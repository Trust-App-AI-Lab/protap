import os
import torch
import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader
import transformers
from transformers import set_seed
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from einops import repeat
import torch.distributed as dist

from models.se3transtormer.se3transformer import SE3Transformer
from models.drug_gvp.drug_gvp import DrugGVPModel
from models.protac.protac_models import Se3ProtacModel
from trainers.trainers import Se3ProtacTrainer, DataCollatorForSe3Protac
from utils.load_models import load_pretrain_model
from utils.metrics import *


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google-t5/t5-base")
    num_labels: int = field(
        default=9,
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="wandb")
    run_name: str = field(default='egnn-pretrain-0310')
    residue_prediction: bool = field(default=False)
    hidden_dim: int = field(default=512)
    max_amino_acids_sequence_length: int = field(default=768)
    mask_ratio: float = field(default=0.15)
    subseq_length: int = field(default=50) # For multi-view contrastive learning.
    temperature: float = field(default=0.01)
    d: float = field(default=15)
    max_nodes: int = field(default=50)
    task: str = field(default='mask_residue_prediction')
    load_pretrain: bool = field(default=True)
    seed: int = field(default=42)
    max_grad_norm: str = field(default=1.0)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        # default="paged_adamw_32bit"
        default='adamw_torch'
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="constant_with_warmup"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    
if __name__ == '__main__':
    os.environ["WANDB_PROJECT"]="protein-pretrain"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    seed = training_args.seed
    set_seed(seed=seed)
    
    print("Loading Dataset...")
    dataset = load_from_disk(data_args.data_path)

    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    # DEBUG
    # dataset = dataset.select(range(0, 96))
    
    print(len(dataset))
    
    if training_args.load_pretrain:
        net = SE3Transformer(
            num_tokens = 22,
            num_positions=training_args.max_amino_acids_sequence_length,
            dim=training_args.hidden_dim,
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
        
        se3 = load_pretrain_model(model_path=model_args.model_name_or_path, model=net)
    else:
        se3 = SE3Transformer(
            num_tokens = 22,
            num_positions=training_args.max_amino_acids_sequence_length,
            dim=training_args.hidden_dim,
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
        
    warhead_ligase_net = DrugGVPModel()
    linker_net = DrugGVPModel()
    
    # torch.autograd.set_detect_anomaly(True)
    model = Se3ProtacModel(
        dim=training_args.hidden_dim * 2 + 128 * 3,
        poi_ligase_model=se3,
        warhead_ligand_model=warhead_ligase_net,
        linker_model=linker_net,
        freeze_encoder=False
    )
        
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    
    trainer = Se3ProtacTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSe3Protac(),
    )
    
    trainer.train()
    # if dist.get_rank() == 0:
    #     torch.save(model.state_dict(), 'egnn_node_pli.pt')
        
    # Evaluation.
    print("Strat evaluation...")
    # modeo = model.to("cuda")
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=96,
        shuffle=True,
        collate_fn=DataCollatorForSe3Protac(),
    )
    yt, yp = [], []
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(test_loader)):
            
            batch_poi_input_ids, batch_e3_ligase_input_ids = inputs['poi_input_ids'], inputs['e3_ligase_input_ids']
            batch_poi_coords, batch_e3_ligase_coords = inputs['poi_coords'], inputs['e3_ligase_coords']
            batch_poi_masks, batch_e3_ligase_masks = inputs['poi_masks'], inputs['e3_ligase_masks']
            batch_warhead, batch_linker, batch_e3_ligand = inputs['warhead'], inputs['linker'], inputs['e3_ligand']
            batch_label = inputs['label']
            
            batch_poi_input_ids = torch.stack(batch_poi_input_ids).to("cuda")
            batch_e3_ligase_input_ids = torch.stack(batch_e3_ligase_input_ids).to("cuda")
            batch_poi_coords = torch.stack(batch_poi_coords).to("cuda")
            batch_e3_ligase_coords = torch.stack(batch_e3_ligase_coords).to("cuda")
            batch_poi_masks = torch.stack(batch_poi_masks).to("cuda")
            batch_e3_ligase_masks = torch.stack(batch_e3_ligase_masks).to("cuda")
            batch_warhead = torch.stack(batch_warhead).to("cuda")
            batch_linker = torch.stack(batch_linker).to("cuda")
            batch_e3_ligand = torch.stack(batch_e3_ligand).to("cuda")
            batch_label = torch.stack(batch_label)
            
            poi_feats = batch_poi_input_ids
            poi_feats = repeat(poi_feats, 'b n -> b (n c)', c=1) # Expand the channel.
            batch_poi_masks = repeat(batch_poi_masks, 'b n -> b (n c)', c=1) # Expand the channel.
            
            i = torch.arange(poi_feats.shape[-1], device=poi_feats.device)
            poi_adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))
            
            e3_ligase_feats = batch_e3_ligase_input_ids
            e3_ligase_feats = repeat(e3_ligase_feats, 'b n -> b (n c)', c=1) # Expand the channel.
            batch_e3_ligase_masks = repeat(batch_e3_ligase_masks, 'b n -> b (n c)', c=1) # Expand the channel.
            
            i = torch.arange(e3_ligase_feats.shape[-1], device=e3_ligase_feats.device)
            e3_ligase_adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

            inputs = {
                "poi_input_ids": poi_feats,
                "poi_coords": batch_poi_coords,
                "poi_masks": batch_poi_masks,
                "e3_ligase_input_ids": e3_ligase_feats,
                "e3_ligase_coords": batch_e3_ligase_coords,
                "e3_ligase_masks": batch_e3_ligase_masks,
                "warhead": batch_warhead,
                "linker": batch_linker,
                "e3_ligand": batch_e3_ligand,
                "label": batch_label,
                "poi_adj_mat" : poi_adj_mat,
                "e3_ligase_adj_mat" : e3_ligase_adj_mat
            }

            logits = model(**inputs)
            prob = torch.softmax(logits, dim=1)[:, 1] # obtain the positive label's probability.
            pred_label = torch.argmax(logits, dim=1)

            yt.append(batch_label.cpu())
            yp.append(prob.cpu())
    
    yt = torch.cat(yt, dim=0).numpy()
    yp = torch.cat(yp, dim=0).numpy()
    
    acc_result = eval_accuray(yt, (yp > 0.5).astype(int))
    auc_result = eval_auc_score(yt, yp)
    
    print("Accuracy:", acc_result)
    print("AUC:", auc_result)
    
    dist.destroy_process_group()
    
    wandb.finish()