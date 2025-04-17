import os

import wandb
import torch.distributed as dist
import transformers
from transformers import set_seed
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional

from models.egnn.egnn import *
from models.drug_gvp.drug_gvp import DrugGVPModel
from models.pli.pli_models import EgnnPLIModel
from trainers.trainers import EgnnPLITrainer, DataCollatorForEgnnPLI
from utils.load_models import load_pretrain_model


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
    fine_tune: bool = field(default=True)
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
    set_seed(42)
    os.environ["WANDB_PROJECT"]="protein-pretrain"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Loading Dataset...")
    dataset = load_from_disk(data_args.data_path)
    # DEBUG
    # dataset = dataset.select(range(0, 96))
    # Rename the column name for training.
    dataset = dataset.rename_column('input_ids', 'feats')
    dataset = dataset.rename_column('coords', 'coors')
    dataset = dataset.rename_column('masks', 'mask')
    
    print(len(dataset))
    
    if training_args.fine_tune:
        net = EGNN_Network(
            num_tokens=22,
            num_positions=training_args.max_amino_acids_sequence_length,  # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=training_args.hidden_dim,
            depth=3,
            num_nearest_neighbors=8,
            coor_weights_clamp_value=2.,   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
        
        egnn = load_pretrain_model(model_path='./checkpoints/egnn_node.pt', model=net)
    else:
        net = EGNN_Network(
            num_tokens=22,
            num_positions=training_args.max_amino_acids_sequence_length,  # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=training_args.hidden_dim,
            depth=3,
            num_nearest_neighbors=8,
            coor_weights_clamp_value=2.,   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
        
    drug_net = DrugGVPModel()
    
    model = EgnnPLIModel(
        dim=2 * training_args.hidden_dim,
        egnn_model=net,
        drug_model=drug_net,
        freeze_egnn=True,
    )

    trainer = EgnnPLITrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForEgnnPLI(),
    )
    
    trainer.train()
    if dist.get_rank() == 0:
        torch.save(net.state_dict(), 'egnn_node_pli.pt')
    
    wandb.finish()