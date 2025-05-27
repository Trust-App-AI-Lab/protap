


from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import math
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.D_transformer.D_transformer import DistanceBertForMLM, GaussianParameters
from models.D_transformer.D_transformer import DistanceBertSelfAttention,DistanceBertLayer,DistanceBertModel

import pickle
from torch.utils.data import Dataset
# ---------------- Hugging-Face imports ----------------
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertModel, BertLayer, BertSelfAttention, BertOnlyMLMHead
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


def gelu(x):
    return x * 0.5 * (1. + torch.erf(x / math.sqrt(2.0)))


AA = list("ARNDCQEGHILKMFPSTWYV")
inv_dist = lambda c: 1.0 / (torch.cdist(c, c, p=2) + 1)

class ProteinDataset(Dataset):
    def __init__(self, pkl, mask_prob=0.15, max_len=1024):
        self.data      = pickle.load(open(pkl, "rb"))
        self.mask_prob = mask_prob
        self.max_len   = max_len
        # token 映射
        self.a2i = {a: i for i, a in enumerate(AA)}
        self.pad = len(AA)       # 20
        self.msk = len(AA) + 1   # 21  [MASK]
        self.unk = len(AA) + 2   # 22  [UNK]
        self.vsz = len(AA) + 3   # 23  vob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e      = self.data[idx]
        ids    = torch.tensor([self.a2i.get(a, self.unk) for a in e["seq"]],
                              dtype=torch.long)
        coords = torch.as_tensor(e["ca_coords"], dtype=torch.float32)
        L      = ids.size(0)
        
        # mask 
        rand = torch.rand(L, device=ids.device)
        mask_arr = (rand < self.mask_prob) & (ids != self.pad) & (ids != self.unk)

        input_ids = ids.clone()
        labels = ids.clone()
        labels[~mask_arr] = -100
        input_ids[mask_arr] = self.msk

        attn = torch.ones(L, dtype=torch.long, device=ids.device)
        dist = inv_dist(coords)

        # max_len
        if L > self.max_len:
            start = torch.randint(0, L - self.max_len + 1, (1,), device=ids.device).item()
            end = start + self.max_len
            input_ids = input_ids[start:end]
            labels    = labels[start:end]
            attn      = attn[start:end]
            dist      = dist[start:end, start:end]

        return input_ids, labels, attn, dist

    @staticmethod
    def collate(batch):
        inps, labs, atts, dists = zip(*batch)
        pad_id = len(AA)
        inps = torch.nn.utils.rnn.pad_sequence(
            inps, batch_first=True, padding_value=pad_id
        )
        labs = torch.nn.utils.rnn.pad_sequence(
            labs, batch_first=True, padding_value=-100
        )
        atts = torch.nn.utils.rnn.pad_sequence(
            atts, batch_first=True, padding_value=0
        )
        B, L = inps.size()
        D = torch.zeros(B, L, L, dtype=dists[0].dtype, device=inps.device)
        for i, d in enumerate(dists):
            l = d.size(0)
            D[i, :l, :l] = d
        return inps, labs, atts, D


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True,         help="Path to pickle file")
    parser.add_argument("--epochs",      type=int, default=50,  help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int, default=32,  help="Batch size per GPU")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_len",     type=int, default=750, help="Maximum sequence length")
    parser.add_argument("--K",           type=int, default=10,   help="Number of Gaussian kernels")
    parser.add_argument("--save_pt", default="D_Trans_mlm_pretrain.pt")

    args = parser.parse_args()
    print(args)

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # --- Dataset & weighting ---
    ds = ProteinDataset(args.data, mask_prob=0.15, max_len=args.max_len)

    # --- DataLoader ---
    sampler = DistributedSampler(ds)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=ProteinDataset.collate,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model, optimizer & scheduler ---
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
    gp = GaussianParameters(args.K)
    base = DistanceBertForMLM(cfg, gp)
    auto_wrap = size_based_auto_wrap_policy
    model = FSDP(base, auto_wrap_policy=auto_wrap,device_id=local_rank,sync_module_states=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    # --- Training loop ---
    for ep in range(1, args.epochs + 1):
        if local_rank == 0:
            print(f"Epoch {ep}: lr = {scheduler.get_last_lr()[0]:.6f}")
        model.train()
        sampler.set_epoch(ep)
        loop = tqdm(dl, desc=f"Epoch {ep}/{args.epochs}", disable=(local_rank != 0))

        for batch_idx, (input_ids, labels, attn_masks, dists) in enumerate(loop, 1):
            # move to device
            input_ids  = input_ids.cuda(non_blocking=True)
            labels     = labels.cuda(non_blocking=True)
            attn_masks = attn_masks.cuda(non_blocking=True)
            dists      = dists.cuda(non_blocking=True)

            # forward + loss
            _, out = model(
                input_ids=input_ids,
                attention_mask=attn_masks,
                distance_matrix=dists
            )
            logits = out["logits"]  # [B, T, V]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # compute mask-token accuracy
            preds    = logits.argmax(dim=-1)
            mask_pos = labels != -100
            if mask_pos.any():
                mask_token_acc = (preds[mask_pos] == labels[mask_pos]).float().mean().item()
            else:
                mask_token_acc = float("nan")
            
            if local_rank == 0:
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{mask_token_acc:.4f}")

        scheduler.step()

    # --- Save full model on rank 0 ---
    with FSDP.summon_full_params(model):
        if dist.get_rank() == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, args.save_pt)
            print(f"Saved full model checkpoint to {args.save_pt}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

