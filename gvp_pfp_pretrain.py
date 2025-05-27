


import pickle
import os, random, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import models.gvp.gvp_model as gvp
import models.gvp.gvp_model.data as gvp_data
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# ─────────────────── utilities ───────────────────
def all_reduce_mean(t: torch.Tensor):
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t

def build_scheduler(opt, warm, total, eta_min=1e-6, start_factor=0.1):
    warmup  = LinearLR(opt, start_factor, 1.0, warm)
    cosine  = CosineAnnealingLR(opt, total-warm, eta_min)
    return SequentialLR(opt, [warmup, cosine], [warm])

# ─────────────────── GVP Encoder ───────────────────
class GVPEncoder(nn.Module):
    def __init__(self, node_dims, edge_dims,
                 num_layers=3, drop_rate=0.1,
                 n_message=3, n_feedforward=2,
                 autoregressive=False):
        super().__init__()
        self.layers = nn.ModuleList([
            gvp.GVPConvLayer(node_dims, edge_dims,
                             n_message=n_message,
                             n_feedforward=n_feedforward,
                             drop_rate=drop_rate,
                             autoregressive=autoregressive)
            for _ in range(num_layers)
        ])
    def forward(self, x, edge_idx, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_idx, edge_attr)
        return x                      # (s,v)

# ─────────────────── Model ───────────────────
class GVP_FP_MODEL(nn.Module):
    def __init__(self,
                 raw_node_s_dim=6,
                 node_v_dim=3,
                 embedding_dim=149,
                 edge_dims=(32,1),
                 num_families=14869):
        super().__init__()
        self.aa_embed = nn.Embedding(22, embedding_dim)
        scalar_dim    = raw_node_s_dim + embedding_dim          # 6 + 149 = 155
        self.encoder  = GVPEncoder((scalar_dim, node_v_dim), edge_dims)
        self.family_table = nn.Embedding(num_families, scalar_dim)
        self.scalar_dim = scalar_dim
    # —— graph encoder  
    def encode_graph(self, node_s, node_v, aa_id,
                     edge_index, edge_s, edge_v, batch_idx):
        aa_emb = self.aa_embed(aa_id)                     # [N,149]
        s_in   = torch.cat([node_s, aa_emb], -1)          # [N,155]
        s_enc,_ = self.encoder((s_in, node_v),
                               edge_index, (edge_s, edge_v))
        return scatter_mean(s_enc, batch_idx, dim=0)      # [B,155]
    # —— forward  
    def forward(self, node_s, node_v, aa_id,
                edge_index, edge_s, edge_v,
                batch_idx,
                fam_idx: torch.Tensor | None = None):
        g_raw = self.encode_graph(node_s, node_v, aa_id,
                                  edge_index, edge_s, edge_v,
                                  batch_idx)
        if fam_idx is None:
            return g_raw

        g_norm  = F.normalize(g_raw, dim=-1)
        f_emb   = self.family_table(fam_idx)                       # [B,K,155]
        f_norm  = F.normalize(f_emb, dim=-1)
        return g_norm, f_norm

# ─────────────────── Loss ───────────────────
def family_contrast_loss(g, f, pos_counts, T=0.01):
    sim = torch.einsum('bd,bkd->bk', g, f) / T
    losses = []
    for i, n_pos in enumerate(pos_counts):
        pos = sim[i, :n_pos]
        neg = sim[i, n_pos:]
        for p in pos:
            logits = torch.cat([p.unsqueeze(0), neg], 0).unsqueeze(0)
            losses.append(F.cross_entropy(logits,
                             torch.zeros(1, device=logits.device,
                                         dtype=torch.long)))
    return torch.stack(losses).mean()

# ─────────────────── collate ───────────────────
def collate(batch): return Batch.from_data_list(batch)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='swiss_540k_list_family_4atom.pkl')
    p.add_argument('--output_dir', default='pfp_weights')
    p.add_argument('--project', default='gvp_pfp_pretrain')
    p.add_argument('--nfam', type=int, default=14869)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--batch_size',  type=int, default=48)
    p.add_argument('--lr',     type=float, default=1e-4)
    p.add_argument('--K',      type=int, default=30)     # pos+neg per sample
    p.add_argument('--temp',   type=float, default=0.01)
    args = p.parse_args()

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    world = dist.get_world_size()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    wandb.login()
    wandb.init(project=args.project, config=vars(args))

    # ---------- data ----------
    if rank == 0: print("Loading dataset …")
    with open(args.data_path, 'rb') as f:
        structure = pickle.load(f)
    
    dataset       = gvp_data.ProteinGraphDataset(structure,family_label=True)
    sampler  = DistributedSampler(dataset, world, rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         sampler=sampler, collate_fn=collate)

    # —— model / optim  
    model = GVP_FP_MODEL(num_families=args.nfam)
    model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy,
                 device_id=rank, sync_module_states=True)

    opt   = optim.Adam(model.parameters(), lr=args.lr)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    # —— epochs  
    for ep in range(args.epochs):
        lr_now = opt.param_groups[0]['lr']
        sampler.set_epoch(ep)
        model.train()

        tot_loss = tot_cnt = 0
        bar = tqdm(loader, desc=f"Ep{ep+1}", disable=(rank!=0))

        for batch in bar:
            batch = batch.to(rank)

            fam_lists, pos_cnts = [], []
            for fam_ids in batch.family:
                fam_ids = list(fam_ids)
                n_pos   = len(fam_ids)
                neg_pool = list(set(range(args.nfam)) - set(fam_ids))
                fam_sel  = fam_ids + random.sample(neg_pool, args.K - n_pos)
                fam_lists.append(fam_sel)
                pos_cnts.append(n_pos)
            fam_tensor = torch.tensor(fam_lists, device=rank)   # [B,K]

            # ----- forward / loss -----  
            g_z, f_z = model(batch.node_s, batch.node_v, batch.seq.clone(),
                             batch.edge_index, batch.edge_s, batch.edge_v,
                             batch.batch, fam_tensor)
            loss = family_contrast_loss(g_z, f_z, pos_cnts, T=args.temp)

            opt.zero_grad()
            loss.backward()
            opt.step()

            reudce_loss = all_reduce_mean(loss.detach())
            tot_loss += reudce_loss.item()
            tot_cnt += 1
            ep_loss = tot_loss / tot_cnt
            if rank == 0:
                bar.set_postfix(loss=f"{reudce_loss.item():.3f}",
                                avg_ls=f"{ep_loss:.3f}",
                                lr=f"{lr_now:.1e}")
        
        sched.step()
        ep_loss = tot_loss / tot_cnt
        if rank == 0:
            print(f"Epoch {ep+1}/{args.epochs} | loss={ep_loss:.3f} | lr={lr_now:.1e}")
            wandb.log({'epoch': ep+1, 'loss': ep_loss, 'lr': lr_now})
        
        with FSDP.summon_full_params(model):
            if dist.get_rank() == 0:
                torch.save(model.state_dict(),os.path.join(args.output_dir,f"family_ep{ep+1}.pt"))

    if rank == 0:
        print("Training finished.")
        wandb.finish()



if __name__ == '__main__':
    main()
