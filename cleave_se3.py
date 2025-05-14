from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import os
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
from einops import repeat

from data.tokenizers import EnzymeTokenizer
from models.se3transtormer.se3transformer import SE3Transformer
from utils.load_models import *

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_and_test_data(train_raw="./data/cleavage_data/train_C14005.pkl",
                        test_raw="./data/cleavage_data/test_C14005.pkl"):

    with open('./data/cleavage_data/Substrate.pickle', 'rb') as fsub:
        sub_protein = pickle.load(fsub)

    data_list = []
    for protein_id, protein_data in sub_protein.items():
        protein_data['name'] = protein_id
        data_list.append(protein_data)

    with open(train_raw, "rb") as ftr:
        train_raw_data = pickle.load(ftr)
    with open(test_raw, "rb") as fte:
        test_raw_data = pickle.load(fte)

    train_label_dict = {k.split("_")[0]: v for k, v in train_raw_data.items()}
    test_label_dict  = {k.split("_")[0]: v for k, v in test_raw_data.items()}

    train_data_list, test_data_list = [], []
    for p in data_list:
        pid = p['name']
        if pid in train_label_dict and len(p["seq"]) >=np.max(train_label_dict[pid]):
            p['cleave_site'] = train_label_dict[pid]
            train_data_list.append(p)
        elif pid in test_label_dict and len(p["seq"]) >=np.max(test_label_dict[pid]):
            p['cleave_site'] = test_label_dict[pid]
            test_data_list.append(p)

    return train_data_list, test_data_list

class CleavageDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        return rec['seq'], rec['coords'], rec['cleave_site']

def collate_fn(batch):
    seqs, coords, lbls = zip(*batch)
    lengths = [len(s) for s in seqs]
    # max_len = max(lengths)
    max_len = 768

    batch_labels = torch.zeros(len(seqs), max_len, dtype=torch.float32)
    for i, positions in enumerate(lbls):
        for p in positions:
            if p < max_len:
                batch_labels[i, p] = 1.0
                
    return list(seqs), coords, batch_labels, torch.tensor(lengths, dtype=torch.long)

class ProteinSE3Encoder(nn.Module):
    def __init__(self,
                #  model_name="facebook/esm2_t33_650M_UR50D",
                se3_model,
                device=torch.device("cuda"),
                freeze_se3: bool=False
                ):
        super().__init__()
        self.device = device
        self.tokenizer = EnzymeTokenizer(max_seq_length=768, padding_to_longest=False)
        self.se3 = se3_model.to(device)
        
        if freeze_se3:
            for param in self.se3.parameters():
                param.requires_grad = False

        self.hidden_size = 36

    def forward(self, batch_seqs, batch_coords):
        input_ids, masks = self.tokenizer.tokenize(batch_seqs)
        batch_input_ids = torch.tensor(input_ids).to(self.device) # [B, L]
        batch_masks = torch.tensor(masks).to(self.device)  # [B, L]

        # Align coords with sequence length
        max_len = batch_input_ids.size(1)
        padded_coords = []

        for coords in batch_coords:
            coords = [c[1] for c in coords]  # Extract [x, y, z] from (atom, [x, y, z])

            if len(coords) > max_len:
                coords = coords[:max_len]
            elif len(coords) < max_len:
                coords += [[0.0, 0.0, 0.0]] * (max_len - len(coords))

            padded_coords.append(coords)

        batch_coords = torch.tensor(padded_coords, dtype=torch.float32).to(self.device)  # [B, L, 3]
        
        feats = batch_input_ids
        feats = repeat(feats, 'b n -> b (n c)', c=1) # Expand the channel.
        batch_masks = repeat(batch_masks, 'b n -> b (n c)', c=1) # Expand the channel.
        
        i = torch.arange(feats.shape[-1], device=feats.device)
        adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))
        
        inputs = {
            "feats" : feats,
            "coors" : batch_coords,
            "mask" : batch_masks.bool(),
            "adj_mat" : adj_mat,
        }
        
        h = self.se3(**inputs)['0'] # (batch_size, max_length, d)
        
        return h

class CleaveSE3Model(nn.Module):
    def __init__(self, se3_encoder,
                 conv_channels=128, kernel_size=31, dropout=0.2):
        super().__init__()
        self.se3 = se3_encoder
        C = 36
        self.conv1d = nn.Conv1d(C, conv_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)
        self.act = nn.SELU()
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels, conv_channels),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, seqs, coords, lengths=None):
        h = self.se3(seqs, coords)              # (B, L, C)
        x = h.permute(0, 2, 1)          # → (B, C, L)
        x = self.drop(self.act(self.conv1d(x)))
        x = x.permute(0, 2, 1)          # → (B, L, conv)
        preds = self.mlp(x).squeeze(-1) # → (B, L)

        return preds

def warmup_cosine_scheduler(
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        start_factor: float = 0.1
):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=int, default=6,
    #                     help='GPU (0,1,2,...)')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_name_or_path', type=str, default='./checkpoints/se3transformer_contrastive.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=24)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    seed_everything(args.seed)

        # device = torch.device(f"cuda:{args.gpu}"
        #                       if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_list, test_list = train_and_test_data(
            train_raw="./data/cleavage_data/train_M10003.pkl",
            test_raw="./data/cleavage_data/test_M10003.pkl"
        )
    train_loader = DataLoader(
            CleavageDataset(train_list),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
    test_loader = DataLoader(
            CleavageDataset(test_list),
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )

    if args.load_pretrain:
        net = SE3Transformer(
            num_tokens = 22,
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
        )
        
        se3 = load_pretrain_model(model_path=args.model_name_or_path, model=net)
    else:
        se3 = SE3Transformer(
            num_tokens = 22,
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
        )
        
    se3_encoder = ProteinSE3Encoder(se3_model=se3, device=device, freeze_se3=args.load_pretrain)
    model = CleaveSE3Model(se3_encoder=se3_encoder).to(rank)
    model = FSDP(model,auto_wrap_policy=size_based_auto_wrap_policy,device_id=rank,sync_module_states=True,use_orig_params=True)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

        # model = nn.DataParallel(model)
        # model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),lr=1e-4
        )
    scheduler = warmup_cosine_scheduler(
        optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.total_epochs,
            min_lr=1e-6,
            start_factor=0.01
        )

    for epoch in range(1, args.total_epochs + 1):
        scheduler.step()
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for seqs, coords, labels, lengths in tqdm(train_loader,
                                              desc=f"Epoch {epoch:02d}"):
            labels = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            preds = model(seqs, coords, lengths)  # (B, L)

            B, L = preds.shape
            mask = (torch.arange(L, device=device)[None, :]
                        < lengths[:, None])
                
            flat_p = preds[mask].detach().cpu().numpy()
            flat_l = labels[mask].detach().cpu().numpy()
            train_preds.extend(flat_p.tolist())
            train_labels.extend(flat_l.tolist())

            loss_mat = F.binary_cross_entropy(preds, labels,
                                                  reduction='none')
            loss = (loss_mat * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        roc_train = roc_auc_score(train_labels, train_preds)
        prec, rec, _ = precision_recall_curve(train_labels, train_preds)
        pr_train = auc(rec, prec)

        print(
                f"Epoch {epoch:02d} | "
                f"loss {train_loss:.4f} | "
                f"Train ROC-AUC {roc_train:.4f} | "
                f"Train AUPR {pr_train:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, coords, labels, lengths in tqdm(test_loader,
                                              desc="Testing"):
            labels = labels.to(device)
            lengths = lengths.to(device)
            preds = model(seqs, coords, lengths).cpu().numpy()
            lengths = lengths.cpu().numpy()

            for i, L in enumerate(lengths):
                all_preds.extend(preds[i, :L].tolist())
                all_labels.extend(labels[i, :L].cpu().numpy().tolist())

    rocauc = roc_auc_score(all_labels, all_preds)
    prec, rec, _ = precision_recall_curve(all_labels, all_preds)
    prauc = auc(rec, prec)

    print(f"Test ROC-AUC: {rocauc:.4f}")
    print(f"Test AUPR:    {prauc:.4f}")


if __name__ == "__main__":
    main()