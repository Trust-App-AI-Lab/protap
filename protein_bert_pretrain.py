import os

import wandb
import torch.distributed as dist
import transformers
from transformers import set_seed
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional

# from models.egnn.egnn import *
from models.proteinbert.proteinbert import *
from trainers.trainers import ProteinBertAttributeMaskingTrainer, ContrastiveProteinBertTrainer, ProteinBertFamilyPredictionTrainer, DataCollatorProteinBertMaskResiduePrediction, DataCollatorProteinBertContrastive, DataCollatorForProteinBertFamilyPrediction


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
    dataset = dataset.rename_column('input_ids', 'seq')
    dataset = dataset.rename_column('coords', 'coors')
    dataset = dataset.rename_column('masks', 'mask')
    
    if training_args.task == 'mask_residue_prediction':
        dataset = dataset.remove_columns('coords') # We do not use coords for masked residue prediction task.
        
    if 'family' in dataset.column_names:
        dataset = dataset.rename_column('family', 'family_labels')
    
    print(len(dataset))
    
    # training_args.max_amino_acids_sequence_length = tokenizer.max_seq_length
    family_prediction = False
    if training_args.task == 'family_prediction':
        family_prediction = True

    model = ProteinBERT(
        num_tokens=22,
        num_annotation=1, # We do not include the GO labels into the training.
        dim=training_args.hidden_dim,
        dim_global=256,
        depth=12,
        narrow_conv_kernel=9,
        wide_conv_kernel=9,
        wide_conv_dilation=5,
        attn_heads=8,
        attn_dim_head=64,
        residue_prediction=training_args.residue_prediction,
        family_prediction=family_prediction,
    )

    if training_args.task == 'mask_node_prediction':
        trainer = ProteinBertAttributeMaskingTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=DataCollatorProteinBertMaskResiduePrediction(),
        )
    elif training_args.task == 'multi_view_contrastive_learning':
        trainer = ContrastiveProteinBertTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=DataCollatorProteinBertContrastive(),
        )
    elif training_args.task == 'family_prediction':
        trainer = ProteinBertFamilyPredictionTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=DataCollatorForProteinBertFamilyPrediction()
        )
    
    trainer.train()
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), 'proteinbert_contrastive.pt')
    
    wandb.finish()