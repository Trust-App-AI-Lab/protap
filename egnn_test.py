import os

import transformers
from transformers import set_seed
from dataclasses import dataclass, field
from typing import Optional

from models.egnn.egnn import *
from data.dataset import EgnnDataset
from data.tokenizers import ProteinTokenizer
from trainers.trainers import AttributeMaskingTrainer, DataCollatorForEgnnMaskResiduePrediction


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
    # report_to: str = field(default="wandb")
    # run_name: str = field(default='llm-fingerprint-1109')
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

    # TODO
    # os.environ["WANDB_PROJECT"]="llm-fingerprint"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    tokenizer = ProteinTokenizer(max_seq_length=256, dataset='egnn-data')
    dataset = EgnnDataset(tokenizer=tokenizer)
    print(dataset[0])

    net = EGNN_Network(
        num_tokens=22,
        num_positions=256,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
        dim=2,
        depth=3,
        num_nearest_neighbors=8,
        coor_weights_clamp_value=2.,   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        residue_prediction=True
    )

    # feats = dataset[0:1]['input_ids']
    # coords = dataset[0:1]['coords']
    # masks = dataset[0:1]['masks']

    # feats_out, coors_out = net(feats, coords, mask = masks) # (1, 1024, 32), (1, 1024, 3)
    # print(feats_out.shape)
    # print(feats_out)
    
    data_collator = DataCollatorForEgnnMaskResiduePrediction()

    trainer = AttributeMaskingTrainer(
        model=net,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    # TODO
    trainer.train()