




import pickle
import os, json, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch_scatter import scatter_mean
from torch_geometric.utils import subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import models.gvp.gvp_model as gvp
import models.gvp.gvp_model.data as gvp_data
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


# -------------------------------------------------- utils
def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Across-GPU mean (scalar tensor)."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t

def warmup_cosine(optimiser, warmup, total, min_lr=1e-6, start_factor=0.1):
    warm  = LinearLR(optimiser, start_factor, 1.0, warmup)
    cos   = CosineAnnealingLR(optimiser, total - warmup, min_lr)
    return SequentialLR(optimiser, [warm, cos], [warmup])

# -------------------------------------------------- model
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
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x                        # (s,v)
    


class GVP_MVCL_MODEL(nn.Module):
    def __init__(self,
                 raw_node_s_dim: int = 6,
                 node_v_dim: int = 3,
                 embedding_dim: int = 149,
                 edge_dims=(32, 1),
                 **enc_kw):
        super().__init__()

        self.aa_embed = nn.Embedding(22, embedding_dim)

        scalar_dim   = raw_node_s_dim + embedding_dim          # 155
        self.encoder = GVPEncoder((scalar_dim, node_v_dim),
                                  edge_dims, **enc_kw)
    def forward(self,
                node_s, node_v, aa_id,
                edge_index, edge_s, edge_v,
                batch_idx):
 
        aa_emb   = self.aa_embed(aa_id)              # [N, 149]
        node_s_  = torch.cat([node_s, aa_emb], -1)   # [N, 155]

        s_enc, _ = self.encoder((node_s_, node_v),
                                edge_index, (edge_s, edge_v))
        # graph-level embedding
        g_enc = scatter_mean(s_enc, batch_idx, dim=0)  # [B, 155]

        return g_enc


def sample_subsequence(data: Data, length=50):
    N = data.seq.size(0)
    if N <= length:
        idx = torch.arange(N, device=data.seq.device)
    else:
        start = random.randint(0, N-length)
        idx = torch.arange(start, start+length, device=data.seq.device)
    ei_sub, _, emask = subgraph(idx, data.edge_index,
                                relabel_nodes=True, num_nodes=N,
                                return_edge_mask=True)
    return Data(
        x=data.x[idx], node_s=data.node_s[idx], node_v=data.node_v[idx],
        seq=data.seq[idx], edge_index=ei_sub,
        edge_s=data.edge_s[emask], edge_v=data.edge_v[emask],
        mask=data.mask[idx] if hasattr(data,"mask") else None,
        pid=getattr(data,"pid", data.name)
    )

def sample_subspace(data: Data, radius=15.0):
    coords = data.x
    if coords.size(0) == 0: return data
    c_idx = torch.randint(0, coords.size(0), (1,)).item()
    dist  = torch.norm(coords - coords[c_idx], dim=-1)
    idx   = (dist <= radius).nonzero(as_tuple=True)[0]
    ei_sub, _, emask = subgraph(idx, data.edge_index,
                                relabel_nodes=True, num_nodes=coords.size(0),
                                return_edge_mask=True)
    return Data(
        x=data.x[idx], node_s=data.node_s[idx], node_v=data.node_v[idx],
        seq=data.seq[idx], edge_index=ei_sub,
        edge_s=data.edge_s[emask], edge_v=data.edge_v[emask],
        mask=data.mask[idx] if hasattr(data,"mask") else None,
        pid=getattr(data,"pid", data.name)
    )

def sample_substructure(data, radius=15.0):
    if random.random() < 0.5:
        return sample_subsequence(data, length=50)
    else:
        return sample_subspace(data, radius=radius)

# -------------------------------------------------- loss (SimCLR)
def info_nce(z, T=0.01):
    """
    z: [2B, dim], (0,1) (2,3) … 为正对
    """
    z = F.normalize(z, dim=-1)                  # cosine similarity
    sim = torch.matmul(z, z.T) / T              # [2B,2B]
    N  = z.size(0)
    pos = torch.arange(N, device=z.device)
    pos[::2] += 1
    pos[1::2] -= 1
    sim.masked_fill_(torch.eye(N, device=z.device).bool(), -9e15)
    return F.cross_entropy(sim, pos)

# -------------------------------------------------- training loop

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',default="swiss_540k_list_family_4atom.pkl")
    p.add_argument('--output_dir', default='mvcl_weights')
    p.add_argument('--project', default='gvp_mvcl_pretrain')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--radius', type=float, default=15.0)
    p.add_argument('--temp', type=float, default=0.01)
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

    ds       = gvp_data.ProteinGraphDataset(structure)
    sampler  = DistributedSampler(ds, world, rank, shuffle=True)
    loader   = DataLoader(ds, batch_size=args.batch_size,
                          sampler=sampler,
                          collate_fn=lambda b: Batch.from_data_list(b))

    # ---------- model ----------
    model = GVP_MVCL_MODEL(embedding_dim=149)
    model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy,
                 device_id=rank, sync_module_states=True)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = warmup_cosine(optimiser, args.warmup_epochs, args.epochs)

    # ---------- epochs ----------
    for ep in range(args.epochs):
        lr_now = optimiser.param_groups[0]['lr']
        sampler.set_epoch(ep)

        total_loss, total_cnt = 0., 0
        bar = tqdm(loader, desc=f"Ep{ep+1}", disable=(rank!=0))

        for batch in bar:
            batch = batch.to(rank)
            sub_list = []
            for data in batch.to_data_list():
                data.pid = getattr(data, "pid", data.name)
                sub1 = sample_substructure(data, radius=args.radius)
                sub2 = sample_substructure(data, radius=args.radius)
                sub_list += [sub1, sub2]
            sub_batch = Batch.from_data_list(sub_list).to(rank)

            # ---- forward ----
            z = model(sub_batch.node_s, sub_batch.node_v, sub_batch.seq.clone(),
                      sub_batch.edge_index, sub_batch.edge_s, sub_batch.edge_v,
                      sub_batch.batch)
            loss = info_nce(z, T=args.temp)

            # ---- optimise ----
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # ---- stats ----
            reduce_loss = reduce_mean(loss.detach())
            total_loss += reduce_loss.item()
            total_cnt  += 1
            epoch_loss = total_loss / total_cnt
            if rank == 0:
                bar.set_postfix(loss=f"{reduce_loss.item():.3f}",avg_loss=f"{epoch_loss:.3f}",lr=f"{lr_now:.1e}")

        scheduler.step()
        epoch_loss = total_loss / total_cnt
        if rank == 0:
            print(f"Epoch {ep+1}/{args.epochs} | loss={epoch_loss:.4f} | lr={lr_now:.1e}")
            wandb.log({"epoch": ep+1, "loss": epoch_loss, "lr": lr_now})
        
        with FSDP.summon_full_params(model):
            if dist.get_rank() == 0:
                torch.save(model.state_dict(),os.path.join(args.output_dir,f"mvcl_epoch_{ep+1}.pt"))
   
    if rank == 0:
        print("Training finished.")
        wandb.finish()


if __name__ == '__main__':
    main()
