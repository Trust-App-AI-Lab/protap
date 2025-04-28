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

from models.egnn.egnn import *
from models.drug_gvp.drug_gvp import DrugGVPModel
from models.pli.pli_models import EgnnPLIModel
from trainers.trainers import EgnnPLITrainer, DataCollatorForEgnnPLI
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
        
    drug_net = DrugGVPModel()
    
    # torch.autograd.set_detect_anomaly(True)
    
    model = EgnnPLIModel(
        dim=training_args.hidden_dim + 128,
        # dim=training_args.hidden_dim,
        egnn_model=egnn,
        drug_model=drug_net,
        # freeze_egnn=training_args.load_pretrain
        freeze_egnn=False
    )
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            
    trainer = EgnnPLITrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForEgnnPLI(),
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
        collate_fn=DataCollatorForEgnnPLI(),
        # num_workers=4
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