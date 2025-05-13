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

from models.proteinbert.proteinbert import *
from models.go.go_models import *
from trainers.trainers import ProteinBertGOTrainer, DataCollatorForProteinBERTGO
from utils.load_models import load_pretrain_model
from utils.metrics import *
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


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
    data_dir = './data/go_data/' + training_args.go_term
    # TODO
    train_set = load_from_disk(data_dir + "_train_2")
    # Rename the column name for training.
    train_set = train_set.rename_column('input_ids', 'seq')
    train_set = train_set.rename_column('masks', 'mask')

    dataset = train_set
    
    test_set = load_from_disk(data_dir + "_test_2")
    # Rename the column name for training.
    test_set = test_set.rename_column('input_ids', 'seq')
    test_set = test_set.rename_column('masks', 'mask')
    # DEBUG
    # dataset = dataset.select(range(0, 96))
    
    print(len(dataset))
    
    if training_args.load_pretrain:
        net = ProteinBERT(
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
        )  
        prot_bert = load_pretrain_model(model_path=model_args.model_name_or_path, model=net)
    else:
        prot_bert = ProteinBERT(
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
        )
    
    model = ProteinBERTGOModel(
        dim=training_args.hidden_dim,
        bert_model=prot_bert,
        go_term=training_args.go_term,
        freeze_bert=training_args.load_pretrain
    )
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            
    trainer = ProteinBertGOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForProteinBERTGO(),
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
        collate_fn=DataCollatorForProteinBERTGO(),
        # num_workers=4
    )
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for step, inputs in tqdm(enumerate(test_loader)):
            
            batch_input_ids, batch_masks, batch_go = inputs['input_ids'], inputs['masks'], inputs['go']
            
            batch_input_ids = torch.stack(batch_input_ids).to('cuda')
            batch_masks = torch.stack(batch_masks).to("cuda")
            batch_go = torch.stack(batch_go)
            
            batch_size = len(batch_input_ids)
            
            annotation = torch.zeros(batch_size, 1, device=batch_input_ids.device)
            
            inputs = {
                "seq" : batch_input_ids,
                "mask" : batch_masks,
                "annotation" : annotation,
            }
            
            logits = model(**inputs)
            probs = torch.sigmoid(logits).cpu()  # shape: [batch_size, num_classes]
            labels = batch_go.cpu()

            all_probs.append(probs)
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    valid_aupr = []
    for i in range(all_labels.shape[1]):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        
        if np.sum(y_true) == 0:
            continue
        
        score = average_precision_score(y_true, y_score)
        valid_aupr.append(score)

    print(f"Macro AUPR over valid classes: {np.mean(valid_aupr)}")
    
    # aupr = average_precision_score(all_labels, all_probs, average='macro')
    # print(f"AUPR: {aupr:.4f}")

    thresholds = np.linspace(0.0, 1.0, num=101)
    fmax = 0.0

    for t in thresholds:
        binarized = (all_probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, binarized, average='macro', zero_division=0)
        fmax = max(fmax, f1)

    print(f"Fmax: {fmax:.4f}")
    
    dist.destroy_process_group()
    
    wandb.finish()