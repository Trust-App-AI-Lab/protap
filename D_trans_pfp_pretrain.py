import argparse
import os
import pickle
import random
from transformers import BertConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from models.D_transformer.D_transformer import DistanceBertForMLM, GaussianParameters
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import pickle
import random
from transformers import BertConfig

AA = list("ARNDCQEGHILKMFPSTWYV")

class ProteinFamilyDataset(Dataset):
    def __init__(self, pkl_file, max_len=750):
        self.data = pickle.load(open(pkl_file, "rb"))
        self.a2i = {a: i for i, a in enumerate(AA)}
        self.pad = len(AA)
        self.unk = len(AA) + 1
        self.vsz = len(AA) + 2
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        seq, coords, family_labels = e['seq'], torch.tensor(e['ca_coords']), e['Pfam']
        L = len(seq)

        if L > self.max_len:
            start = random.randint(0, L - self.max_len)
            seq = seq[start:start + self.max_len]
            coords = coords[start:start + self.max_len]

        input_ids = torch.full((self.max_len,), self.pad, dtype=torch.long)
        attn = torch.zeros(self.max_len, dtype=torch.long)
        dist_matrix = torch.zeros((self.max_len, self.max_len), dtype=torch.float)

        seq_ids = [self.a2i.get(a, self.unk) for a in seq]
        input_ids[:len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long)
        attn[:len(seq_ids)] = 1

        dist_matrix[:len(seq_ids), :len(seq_ids)] = 1.0 / (torch.cdist(coords, coords) + 1)

        return input_ids, attn, dist_matrix, family_labels

    @staticmethod
    def collate(batch):
        input_ids, attn, dist_matrix, family_labels = zip(*batch)
        input_ids = torch.stack(input_ids)
        attn = torch.stack(attn)
        dist_matrix = torch.stack(dist_matrix)
        return input_ids, attn, dist_matrix, family_labels

class FamilyDistanceBert(nn.Module):
    def __init__(self, cfg, gp, num_families):
        super().__init__()
        self.model = DistanceBertForMLM(cfg, gp)
        self.family_embedding = nn.Embedding(num_families, cfg.hidden_size)

    def forward(self, input_ids, attn, dist_matrix, family_labels):
        outputs, _ = self.model(input_ids, attn, distance_matrix=dist_matrix)
        pooled_output = (outputs * attn.unsqueeze(-1)).sum(1) / attn.sum(1, keepdim=True)
        family_emb = self.family_embedding(family_labels)
        return pooled_output, family_emb

def info_nce_loss(z, f, temperature=0.01):
    z, f = F.normalize(z, dim=-1), F.normalize(f, dim=-1)
    logits = torch.einsum('bd,bkd->bk', z, f) / temperature
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)

def train(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    ds = ProteinFamilyDataset(args.data, max_len=args.max_len)
    sampler = DistributedSampler(ds, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=ds.collate)

    cfg = BertConfig(
        vocab_size=ds.vsz,
        pad_token_id=ds.pad,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=args.max_len,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    gp = GaussianParameters(K=args.K)
    model = FamilyDistanceBert(cfg, gp, num_families=args.num_families)
    model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy, device_id=rank, sync_module_states=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=5),
                    CosineAnnealingLR(optimizer, T_max=args.epochs-5, eta_min=1e-6)],
        milestones=[5])

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        batch_count =0
        if rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = loader

        for input_ids, attn, dist_matrix, family_labels in pbar:
            input_ids, attn, dist_matrix = [x.cuda(rank) for x in (input_ids, attn, dist_matrix)]

            sampled_family_labels = []
            for labels in family_labels:
                pos_labels = labels
                neg_pool = set(range(args.num_families)) - set(pos_labels)
                sampled_neg = random.sample(list(neg_pool), 30 - len(pos_labels))
                sampled_family_labels.append(pos_labels + sampled_neg)

            sampled_family_labels = torch.tensor(sampled_family_labels, device=rank)

            z, f = model(input_ids, attn, dist_matrix, sampled_family_labels)
            loss = info_nce_loss(z, f)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            avg_loss = total_loss / batch_count
            if rank == 0:
                pbar.set_postfix({
                    "loss":    f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })
        scheduler.step()


    with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
        if dist.get_rank() == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, args.save_pt)
            print(f"Saved full model checkpoint to {args.save_pt}")


    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pickle file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=750)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--num_families", type=int, default=14869)
    parser.add_argument("--save_pt", default="D_Trans_pfp_pretrain.pt")
    args = parser.parse_args()
    train(args)
