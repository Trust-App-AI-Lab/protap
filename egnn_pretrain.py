import os

import wandb
import torch.distributed as dist
import transformers
from transformers import set_seed
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional

from models.egnn.egnn import *
from trainers.trainers import AttributeMaskingTrainer, ContrastiveEGNNTrainer, DataCollatorForEgnnMaskResiduePrediction


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
    
    # training_args.max_amino_acids_sequence_length = tokenizer.max_seq_length

    net = EGNN_Network(
        # num_tokens=tokenizer.amino_numbers,
        num_tokens=22,
        num_positions=training_args.max_amino_acids_sequence_length,  # unless what you are passing in is an unordered set, set this to the maximum sequence length
        dim=training_args.hidden_dim,
        depth=3,
        num_nearest_neighbors=8,
        coor_weights_clamp_value=2.,   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        residue_prediction=training_args.residue_prediction,
    )
    
    data_collator = DataCollatorForEgnnMaskResiduePrediction()

    if training_args.task == 'mask_node_prediction':
        trainer = AttributeMaskingTrainer(
            model=net,
            train_dataset=dataset,
            args=training_args,
            data_collator=data_collator,
        )
    elif training_args.task == 'multi_view_contrastive_learning':
        trainer = ContrastiveEGNNTrainer(
            model=net,
            train_dataset=dataset,
            args=training_args,
            data_collator=DataCollatorForEgnnMaskResiduePrediction()
        )
    
    trainer.train()
    if dist.get_rank() == 0:
        torch.save(net.state_dict(), 'egnn_contrastive.pt')
    
    wandb.finish()