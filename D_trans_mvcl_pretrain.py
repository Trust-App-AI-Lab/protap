import argparse
import os
import pickle
import random
from transformers import BertConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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

import warnings
warnings.filterwarnings("ignore")


AA = list("ARNDCQEGHILKMFPSTWYV")

def inv_dist(c):
    return 1.0 / (torch.cdist(c, c, p=2) + 1)


def warmup_cosine_scheduler(
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        start_factor: float = 0.1
):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler


class ContrastiveProteinDataset(Dataset):
    def __init__(self, pkl_file, max_len=750, sub_len=50, sub_radius=15):
        self.data = pickle.load(open(pkl_file, "rb"))
        self.a2i = {a: i for i, a in enumerate(AA)}
        self.pad = len(AA)
        self.unk = len(AA) + 1
        self.vsz = len(AA) + 2
        self.max_len = max_len
        self.sub_len = sub_len
        self.sub_radius = sub_radius

    def __len__(self):
        return len(self.data)

    def subsequence_sampling(self, seq_len):
        if seq_len <= self.sub_len:
            return torch.arange(seq_len)
        start = random.randint(0, seq_len - self.sub_len)
        return torch.arange(start, start + self.sub_len)

    def subspace_sampling(self, coords):
        center_idx = random.randint(0, coords.size(0) - 1)
        center_coord = coords[center_idx]
        distances = torch.norm(coords - center_coord, dim=-1)
        return torch.where(distances <= self.sub_radius)[0]

    def create_subsample(self, seq, coords):
        if random.random() < 0.5:
            idx = self.subsequence_sampling(len(seq))
        else:
            idx = self.subspace_sampling(coords)

        input_ids = torch.full((self.max_len,), self.pad, dtype=torch.long)
        attn = torch.zeros(self.max_len, dtype=torch.long)
        dist_matrix = torch.zeros((self.max_len, self.max_len), dtype=torch.float)

        selected_seq = [self.a2i.get(seq[i], self.unk) for i in idx]
        input_ids[:len(idx)] = torch.tensor(selected_seq, dtype=torch.long)
        attn[:len(idx)] = 1

        sampled_coords = coords[idx]
        sampled_dist = inv_dist(sampled_coords)
        dist_matrix[:len(idx), :len(idx)] = sampled_dist

        return input_ids, attn, dist_matrix

    def __getitem__(self, idx):
        e = self.data[idx]
        seq, coords = e['seq'], torch.tensor(e['ca_coords'])
        L = len(seq)

        if L > self.max_len:
            start = random.randint(0, L - self.max_len)
            seq = seq[start:start + self.max_len]
            coords = coords[start:start + self.max_len]

        sample1 = self.create_subsample(seq, coords)
        sample2 = self.create_subsample(seq, coords)

        return (*sample1, *sample2)

    @staticmethod
    def collate(batch):
        ids1, attn1, dist1, ids2, attn2, dist2 = zip(*batch)
        ids1 = torch.stack(ids1)
        attn1 = torch.stack(attn1)
        dist1 = torch.stack(dist1)
        ids2 = torch.stack(ids2)
        attn2 = torch.stack(attn2)
        dist2 = torch.stack(dist2)

        return (ids1, attn1, dist1), (ids2, attn2, dist2)

def info_nce_loss(z1, z2, temperature=0.01):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    labels = torch.arange(z1.size(0)).cuda()
    similarity_matrix = torch.matmul(z1, z2.T) / temperature

    loss = (F.cross_entropy(similarity_matrix, labels) +
            F.cross_entropy(similarity_matrix.T, labels)) / 2

    return loss

def mean_pooling(hidden_states, attn):
    attn_expanded = attn.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * attn_expanded, 1)
    sum_mask = torch.clamp(attn_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def warmup_cosine_scheduler(
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        start_factor: float = 0.1
):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler



def train_fsdp(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    ds = ContrastiveProteinDataset(args.data, max_len=args.max_len)
    sampler = DistributedSampler(ds,shuffle=True)
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
    base_model = DistanceBertForMLM(cfg, gp)
    model = FSDP(base_model,auto_wrap_policy=size_based_auto_wrap_policy,device_id=rank,sync_module_states=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine_scheduler(optim, warmup_epochs=5, total_epochs=args.epochs)


    for ep in range(args.epochs):
        model.train()
        sampler.set_epoch(ep)
        total = 0.0
        batch_count =0
        if rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {ep+1}/{args.epochs}")
        else:
            pbar = loader
        for (i1, a1, d1), (i2, a2, d2) in pbar:
            i1, a1, d1, i2, a2, d2 = [x.cuda() for x in (i1, a1, d1, i2, a2, d2)]
            h1, _ = model(i1, a1, distance_matrix=d1)
            h2, _ = model(i2, a2, distance_matrix=d2)
            z1, z2 = mean_pooling(h1, a1), mean_pooling(h2, a2)
            loss = info_nce_loss(z1, z2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            batch_count += 1
            avg_loss = total / batch_count
            if rank == 0:
                pbar.set_postfix({
                    "loss":    f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })
        sched.step()
        if rank == 0:
            print(f"Epoch {ep+1}, Avg Loss: {total/len(loader):.4f}")

    # --- Save full model on rank 0 ---
    with FSDP.summon_full_params(model):
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), args.save_pt)
            print(f"Saved full model checkpoint to {args.save_pt}")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,          help="Path to pickle file")
    parser.add_argument("--epochs",     type=int, default=50,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,   help="Batch size per GPU")
    parser.add_argument("--lr",         type=float, default=1e-4,help="Learning rate")
    parser.add_argument("--max_len",    type=int, default=750,  help="Maximum sequence length")
    parser.add_argument("--K",          type=int, default=10,   help="Number of Gaussian kernels")
    parser.add_argument("--save_pt",       default="D_Trans_mvcl_pretrain.pt", help="Path to save checkpoint")
    args = parser.parse_args()
    train_fsdp(args)
