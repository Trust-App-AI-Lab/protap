import os

import wandb
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
import transformers
from transformers import set_seed
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch.distributed as dist

from models.egnn.egnn import *
from models.cleavage_site.cleavage_site_models import *
from trainers.trainers import EgnnCleavageTrainer, DataCollatorForEgnnCleavage
from utils.load_models import load_pretrain_model
from utils.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score

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
    go_term: str = field(default='biological_process')
    enzyme: str = field(default='c14005')
    alpha: float = field(default=0.25)
    gamma: float = field(default=2.0)
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
    data_dir = './data/cleavage_data/' + training_args.enzyme
    # TODO
    train_set = load_from_disk(data_dir + "_train_3")
    # Rename the column name for training.
    train_set = train_set.rename_column('input_ids', 'feats')
    train_set = train_set.rename_column('coords', 'coors')
    train_set = train_set.rename_column('masks', 'mask')

    dataset = train_set
    
    test_set = load_from_disk(data_dir + "_test_3")
    # Rename the column name for training.
    test_set = test_set.rename_column('input_ids', 'feats')
    test_set = test_set.rename_column('coords', 'coors')
    test_set = test_set.rename_column('masks', 'mask')
    # DEBUG
    # dataset = dataset.select(range(0, 96))
    
    print(len(dataset))
    
    if training_args.load_pretrain:
        net = EGNN_Network(
            num_tokens=22,
            num_positions=training_args.max_amino_acids_sequence_length,  # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=training_args.hidden_dim,
            depth=3,
            num_nearest_neighbors=8,
            coor_weights_clamp_value=2.,   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
        
        egnn = load_pretrain_model(model_path=model_args.model_name_or_path, model=net)
    else:
        egnn = EGNN_Network(
            num_tokens=22,
            num_positions=training_args.max_amino_acids_sequence_length,  # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=training_args.hidden_dim,
            depth=3,
            num_nearest_neighbors=8,
            coor_weights_clamp_value=2.,   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
    
    model = EgnnCleavageModel(
        dim=training_args.hidden_dim,
        egnn_model=egnn,
        freeze_egnn=training_args.load_pretrain
    )
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            
    trainer = EgnnCleavageTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForEgnnCleavage(),
    )
    
    trainer.train()
    # if dist.get_rank() == 0:
    #     torch.save(model.state_dict(), 'egnn_node_pli.pt')
        
    # Evaluation.
    print("Strat evaluation...")
    # modeo = model.to("cuda")
    model.eval()
    test_loader = DataLoader(
        test_set,
        batch_size=96,
        shuffle=True,
        collate_fn=DataCollatorForEgnnCleavage(),
        # num_workers=4
    )
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for step, inputs in tqdm(enumerate(test_loader)):
            
            batch_input_ids, batch_coords, batch_masks, batch_site = inputs['input_ids'], inputs['coords'], inputs['masks'], inputs['site']
        
            batch_input_ids = torch.stack(batch_input_ids).to("cuda")
            batch_coords = torch.stack(batch_coords).to("cuda")
            batch_masks = torch.stack(batch_masks).to("cuda")
            batch_site = torch.stack(batch_site)

            inputs = {
                "feats" : batch_input_ids,
                "coors" : batch_coords,
                "mask" : batch_masks,
                "site" : batch_site
            }
            
            logits = model(**inputs)
            probs = torch.sigmoid(logits).cpu()  # shape: [batch_size, num_classes]
            labels = batch_site.cpu()

            all_probs.append(probs)
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    
    valid_scores = []

    for i in range(all_labels.shape[1]):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        if np.sum(y_true) == 0:
            continue

        try:
            auc = roc_auc_score(y_true, y_score)
            valid_scores.append(auc)
        except ValueError as e:
            print(f"Skipping class {i} due to error: {e}")
            
    mean_auc = np.mean(valid_scores)
    print(f"Macro ROC AUC over valid classes: {mean_auc}")
    
    valid_aupr = []
    for i in range(all_labels.shape[1]):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        
        if np.sum(y_true) == 0:
            continue
        
        score = average_precision_score(y_true, y_score)
        valid_aupr.append(score)

    print(f"Macro AUPR over valid classes: {np.mean(valid_aupr)}")
    
    # aupr = average_precision_score(all_labels, all_probs)
    # print(f"AUPR: {aupr}")
    
    dist.destroy_process_group()
    
    wandb.finish()