
import os, argparse, pickle, torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import wandb

import models.gvp.gvp_model as gvp
import models.gvp.gvp_model.data as gvp_data


# -------------- util --------------
def reduce_loss(tensor: torch.Tensor):
    """Across-GPU mean."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                            min_lr=1e-6, start_factor=0.1):
    warmup  = LinearLR(optimizer, start_factor, 1.0, warmup_epochs)
    cosine  = CosineAnnealingLR(optimizer, total_epochs - warmup_epochs, min_lr)
    return SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])

# -------------- model --------------
class GVPEncoder(nn.Module):
    def __init__(self, node_dims, edge_dims, num_layers=3,
                 drop_rate=0.1, n_message=3, n_feedforward=2,
                 autoregressive=False):
        super().__init__()
        self.layers = nn.ModuleList([
            gvp.GVPConvLayer(node_dims, edge_dims, n_message,
                             n_feedforward, drop_rate, autoregressive)
            for _ in range(num_layers)
        ])
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

class GVP_MLM_MODEL(nn.Module):
    def __init__(self, raw_node_s_dim, node_v_dim, embedding_dim,
                 edge_dims=(32,1), **enc_kw):
        super().__init__()
        self.aa_embed   = nn.Embedding(22, embedding_dim)   # 0-19 aa, 20 mask
        total_s_dim     = raw_node_s_dim + embedding_dim
        self.encoder    = GVPEncoder((total_s_dim, node_v_dim),
                                     edge_dims, **enc_kw)
        self.classifier = nn.Linear(total_s_dim, 20)
    def forward(self, node_s, node_v, aa_id,
                edge_index, edge_s, edge_v, mask_idx=None):
        aa = aa_id.clone()
        if mask_idx is not None:
            aa[mask_idx] = 20            # mask token
        aa_emb  = self.aa_embed(aa)
        node_s_ = torch.cat([node_s, aa_emb], -1)
        s_enc,_ = self.encoder((node_s_, node_v), edge_index, (edge_s, edge_v))
        return self.classifier(s_enc)    # logits

def collate_fn(b): return Batch.from_data_list(b)

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='swiss_540k_list_family_4atom.pkl')
    ap.add_argument('--output_dir', default='mlm_weights')
    ap.add_argument('--project', default='gvp_mlm_pretrain')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--embedding_dim', type=int, default=149)
    ap.add_argument('--raw_node_s_dim', type=int, default=6)
    ap.add_argument('--node_v_dim', type=int, default=3)
    ap.add_argument('--mask_ratio', type=float, default=0.15)
    args = ap.parse_args()

    print(args)

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    world = dist.get_world_size()

    if local_rank == 0: os.makedirs(args.output_dir, exist_ok=True)

    wandb.login()
    wandb.init(project=args.project, config=vars(args))

    # -------- dataset --------
    with open(args.data_path, 'rb') as f:
        ds = gvp_data.ProteinGraphDataset(pickle.load(f))

    sampler  = DistributedSampler(ds, world, local_rank, shuffle=True)
    loader   = DataLoader(ds, batch_size=args.batch_size,
                          sampler=sampler, collate_fn=collate_fn)

    # -------- model/opt --------
    model = GVP_MLM_MODEL(
        raw_node_s_dim=args.raw_node_s_dim,
        node_v_dim=args.node_v_dim,
        embedding_dim=args.embedding_dim
    )
    model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy,
                 device_id=local_rank, sync_module_states=True)

    opt   = optim.Adam(model.parameters(), lr=args.lr)
    sched = warmup_cosine_scheduler(opt, args.warmup_epochs, args.epochs)

    # -------- training --------
    for epoch in range(args.epochs):
        sched.step()
        lr_now = opt.param_groups[0]['lr']
        sampler.set_epoch(epoch)

        total_loss, total_tok = 0., 0
        bar = tqdm(loader, desc=f'Ep{epoch+1}', disable=(local_rank!=0),
                   dynamic_ncols=True)

        for batch in bar:
            batch = batch.to(local_rank)

            # ----- mask -----
            if hasattr(batch, 'mask'):
                valid = torch.nonzero(batch.mask, as_tuple=True)[0]
            else:
                valid = torch.arange(len(batch.seq), device=local_rank)
            mcnt  = max(1, int(args.mask_ratio * valid.numel()))
            midx  = valid[torch.randperm(valid.numel(), device=local_rank)[:mcnt]]

            # ----- forward & loss -----
            logits = model(batch.node_s, batch.node_v, batch.seq.clone(),
                           batch.edge_index, batch.edge_s, batch.edge_v,
                           mask_idx=midx)
            loss = F.cross_entropy(logits[midx], batch.seq[midx].clamp(max=19))

            # ----- backward -----
            opt.zero_grad()
            loss.backward()
            opt.step()

            # ----- all-reduce & stats -----
            loss_reduced = reduce_loss(loss.detach())
            step_loss    = loss_reduced.item()
            tok_this     = midx.numel()
            total_loss  += step_loss * tok_this
            total_tok   += tok_this
            avg_loss     = total_loss / total_tok

            if local_rank == 0:
                bar.set_postfix(lr=f'{lr_now:.1e}',
                                step=f'{step_loss:.4f}',
                                avg=f'{avg_loss:.4f}')
                wandb.log(dict(
                    epoch = epoch+1,
                    lr    = lr_now,
                    step_loss = step_loss,
                    avg_loss  = avg_loss,
                    global_step = epoch*len(loader)+bar.n
                ))

        # ------ epoch summary ------
        if local_rank == 0:
            epoch_avg = total_loss / total_tok
            print(f'Epoch {epoch+1} done | avg_loss={epoch_avg:.4f} | lr={lr_now:.1e}')
            wandb.log({'epoch_avg_loss': epoch_avg, 'lr_epoch': lr_now})
        
        with FSDP.summon_full_params(model):
            if dist.get_rank() == 0:
                torch.save(model.state_dict(),os.path.join(args.output_dir,f'mask_pretrain_epoch_{epoch+1}.pt'))

    if local_rank == 0:
        print('Training complete.')
        wandb.finish()


if __name__ == '__main__':
    main()

