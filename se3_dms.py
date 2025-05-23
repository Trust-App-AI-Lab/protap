import os
import math
import pickle
import warnings
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random
from einops import repeat
from models.se3transtormer.se3transformer import SE3Transformer
from data.tokenizers import EnzymeTokenizer
from utils.load_models import load_pretrain_model

warnings.filterwarnings("ignore")


DMS_DIR = "./data/mutation_effects/DMS_ProteinGym_substitutions"
SAVE_DIR = "./data/mutation_effects/SE3_DMS_results"
STRUCT_PATH = "./data/mutation_effects/DMS_structure.pkl"

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOKEN2ID = {
 'D': 0, 'F': 1, 'H': 2, 'N': 3, 'A': 4, 'I': 5, 'V': 6, 'M': 7,
 'E': 8, 'G': 9, 'K': 10, 'Q': 11, 'C': 12, 'W': 13, 'Y': 14,
 'P': 15, 'S': 16, 'L': 17, 'T': 18, 'R': 19, '<MASK>': 20, '<PAD>': 21
}
MASK_ID = 20
MAX_LEN = 768
BATCH_SIZE = 64

tokenizer = EnzymeTokenizer(max_seq_length=MAX_LEN, padding_to_longest=False)

net = SE3Transformer(
    num_tokens=22,
    num_positions=768,
    dim=36,
    dim_head = 8,
    heads = 2,
    depth = 2,
    attend_self = True,
    input_degrees = 1,
    output_degrees = 2,
    reduce_dim_out = False,
    differentiable_coors = True,
    num_neighbors = 0,
    attend_sparse_neighbors = True,
    num_adj_degrees = 2,
    adj_dim = 4,
    num_degrees=2,
    residue_prediction=True
)

model = load_pretrain_model(model_path='./checkpoints/se3transformer_node.pt', model=net).to(device)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model).to(device)
# model.eval()

struct_db = pickle.load(open(STRUCT_PATH, "rb"))

def get_ca_coords(ca_list):
    if len(ca_list) >= MAX_LEN:
        ca = ca_list[:MAX_LEN]
    else:
        ca = ca_list + [[0., 0., 0.]] * (MAX_LEN - len(ca_list))
    return torch.tensor(ca, dtype=torch.float32, device=device)  # (L,3)

@torch.no_grad()
def predict_batch(seqs, infos, ca_tensor):
    N = len(seqs)
    scores = [float("nan")] * N

    legal = []
    for i, info in enumerate(infos):
        if max(int(m[1:-1]) for m in info.split(":")) <= MAX_LEN:
            legal.append(i)
    if not legal:
        return scores

    step = min(BATCH_SIZE, len(legal))
    for s in tqdm(range(0, len(legal), step), leave=False, desc="批次"):
        idxs  = legal[s:s+step]
        bs    = [seqs[i]  for i in idxs]
        infos_b = [infos[i] for i in idxs]

        ids, masks = tokenizer.tokenize(bs)
        feats = torch.tensor(ids, dtype=torch.long,  device=device)  # (B,L)
        mask = torch.tensor(masks, dtype=torch.bool, device=device)   # (B,L)
        B, L = feats.shape

        coors = ca_tensor.unsqueeze(0).expand(B, -1, -1) # (B,L,3)

        pos_list, ori_ids, mut_ids = [], [], []
        feats_masked = feats.clone()
        for b, info in enumerate(infos_b):
            p_lst, o_lst, m_lst = [], [], []
            for mut in info.split(":"):
                ori, pos, mut_aa = mut[0], int(mut[1:-1])-1, mut[-1]
                p_lst.append(pos)
                o_lst.append(TOKEN2ID[ori])
                m_lst.append(TOKEN2ID[mut_aa])
                feats_masked[b, pos] = MASK_ID
            pos_list.append(p_lst)
            ori_ids.append(o_lst)
            mut_ids.append(m_lst)
        
        feats_masked = repeat(feats_masked, 'b n -> b (n c)', c=1) # Expand the channel.
        mask = repeat(mask, 'b n -> b (n c)', c=1) # Expand the channel.
        
        i = torch.arange(feats_masked.shape[-1], device=feats_masked.device)
        adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

        logits = model(feats=feats_masked, coors=coors, mask=mask, adj_mat=adj_mat)['0']
        # print(logits)# (B,L,22)
        # print(logits.shape)
        probs  = F.softmax(logits, dim=-1).cpu()

        for k, b in enumerate(range(B)):
            sc = 0.0
            for j, p in enumerate(pos_list[b]):
                sc += math.log((probs[b, p, mut_ids[b][j]] + 1e-9) /
                               (probs[b, p, ori_ids[b][j]] + 1e-9))
            scores[idxs[k]] = sc
    return scores

csv_files = [f for f in os.listdir(DMS_DIR) if f.endswith(".csv")]

file_length = {}
for i in os.listdir(DMS_DIR):
    df = pd.read_csv(os.path.join(DMS_DIR, i))
    file_length[i] = df.shape[0]

small_files = [k for k, v in file_length.items() if v < 20000]
print(len(csv_files))
print(len(small_files))

evaluate_files = random.sample(small_files, 40)

for fname in tqdm(evaluate_files, desc="file"):
    out_path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(out_path):
        continue

    key = fname.split(".")[0]
    if key not in struct_db:
        continue

    ca_list = [res[1] for res in struct_db[key]["coords"]]
    ca_tensor = get_ca_coords(ca_list)

    df    = pd.read_csv(os.path.join(DMS_DIR, fname))
    seqs  = df["mutated_sequence"].tolist()
    infos = df["mutant"].tolist()

    df["predict_effect"] = predict_batch(seqs, infos, ca_tensor)
    df.to_csv(out_path, index=False)

print("Complete")