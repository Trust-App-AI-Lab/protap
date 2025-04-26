import os

import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader
import transformers
from transformers import set_seed
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from models.proteinbert.proteinbert import *
from models.drug_gvp.drug_gvp import DrugGVPModel
from models.pli.pli_models import ProteinBertPLIModel
from trainers.trainers import ProteinBertPLITrainer, DataCollatorForProteinBERTPLI
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
    # TODO
    dataset = load_from_disk(data_args.data_path)
    # Rename the column name for training.
    dataset = dataset.rename_column('input_ids', 'seq')
    # dataset = dataset.remove_column('coords', 'coors')
    dataset = dataset.rename_column('masks', 'mask')
    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    dataset = split_dataset['train']
    test_dataset = split_dataset['test']
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
        
    drug_net = DrugGVPModel()
    
    # torch.autograd.set_detect_anomaly(True)
    
    model = ProteinBertPLIModel(
        dim=training_args.hidden_dim + 128,
        # dim=training_args.hidden_dim,
        proteinbert_model=prot_bert,
        drug_model=drug_net,
        # freeze_egnn=training_args.load_pretrain
        freeze_bert=True
    )
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            
    trainer = ProteinBertPLITrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForProteinBERTPLI(),
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
        collate_fn=DataCollatorForProteinBERTPLI(),
        # num_workers=4
    )
    yt, yp = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(test_loader)):
            
            batch_input_ids, batch_masks, batch_drugs, batch_y = inputs['input_ids'], inputs['masks'], inputs['drugs'], inputs['y']
        
            batch_input_ids = torch.stack(batch_input_ids).to("cuda")
            batch_masks = torch.stack(batch_masks).to("cuda")
            batch_drugs = torch.stack(batch_drugs).to("cuda")
            batch_y = torch.stack(batch_y)
            
            batch_size = len(batch_input_ids)

            annotation = torch.zeros(batch_size, 1, device=batch_input_ids.device)
            
            inputs = {
                "seq" : batch_input_ids,
                "mask" : batch_masks,
                "annotation" : annotation,
                "drugs" : batch_drugs
            }
            
            y = batch_y
            yh = model(**inputs)
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    
    mse_result = eval_mse(yt, yp)
    pearson_result = eval_pearson(yt, yp)
    
    print("MSE:", mse_result['mse'])
    print("Pearson r:", pearson_result['pearsonr'])
    
    dist.destroy_process_group()
    
    wandb.finish()