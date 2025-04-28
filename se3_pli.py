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
import torch.distributed as dist

from models.se3transtormer.se3transformer import *
from models.drug_gvp.drug_gvp import DrugGVPModel
from models.pli.pli_models import Se3PLIModel
from trainers.trainers import Se3PLITrainer, DataCollatorForSe3PLI
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
    dataset = dataset.rename_column('input_ids', 'feats')
    dataset = dataset.rename_column('coords', 'coors')
    dataset = dataset.rename_column('masks', 'mask')
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
        
    drug_net = DrugGVPModel()
    
    model = Se3PLIModel(
        dim=training_args.hidden_dim + 128,
        se3_model=se3,
        drug_model=drug_net,
        freeze_se3=False
    )
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            
    trainer = Se3PLITrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSe3PLI(),
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
        collate_fn=DataCollatorForSe3PLI(),
    )
    yt, yp = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(test_loader)):
            
            batch_input_ids, batch_coords, batch_masks, batch_drugs, batch_y = inputs['input_ids'], inputs['coords'], inputs['masks'], inputs['drugs'], inputs['y']
        
            batch_input_ids = torch.stack(batch_input_ids).to("cuda")
            batch_coords = torch.stack(batch_coords).to("cuda")
            batch_masks = torch.stack(batch_masks).to("cuda")
            batch_drugs = torch.stack(batch_drugs).to("cuda")
            batch_y = torch.stack(batch_y)

            inputs = {
                "feats" : batch_input_ids,
                "coors" : batch_coords,
                "mask" : batch_masks,
                "drugs" : batch_drugs,
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