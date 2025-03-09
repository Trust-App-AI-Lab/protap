import transformers
from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass

from typing import Sequence, Dict

EGNN_MASK_TOKEN = 20
EGNN_PAD_TOKEN = 21


@dataclass
class DataCollatorForEgnnMaskResiduePrediction(object):
    """Collate examples for training EGNN with Maksed Residue Prediction task."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, coords, masks = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "coords", "masks")
        )
        
        return dict(
            input_ids=input_ids,
            coords=coords,
            masks=masks,
        )

class AttributeMaskingTrainer(Trainer):
    """
    Attribute masking trainer using Hugging Face Trainer framework for pretraining graph neural networks.

    Parameters:
        model (nn.Module): Node representation model
        mask_rate (float, optional): Rate of masked nodes
        num_mlp_layer (int, optional): Number of MLP layers
        graph_construction_model (optional): Graph construction model for enhancing graph features
    """

    def compute_loss(
        self,
        model,
        inputs,
    ):
        """
        Compute loss for a batch using cross entropy loss.
        """
        batch_input_ids, batch_coords, batch_masks = inputs['input_ids'], inputs['coords'], inputs['masks']
        batch_size, seq_len = batch_input_ids.shape
        new_mask = batch_masks.clone()
        target = torch.full_like(batch_input_ids, -100)

        for i in range(batch_size):
            non_pad_indices = (batch_input_ids[i] != EGNN_PAD_TOKEN).nonzero(as_tuple=True)[0]
            num_to_mask = max(1, int(len(non_pad_indices) * self.args.mask_ratio))
            mask_indices = torch.randperm(len(non_pad_indices))[:num_to_mask] # (num_to_mask, )
            
            selected_indices = non_pad_indices[mask_indices]
            new_mask[i, selected_indices] = False  # Masked residues
            target[i, selected_indices] = batch_input_ids[selected_indices]  # Assign the masked reidues with their labels.
        
        pred = model(**inputs) # (batch_size, seq_length, 21)

        loss = F.cross_entropy(pred, target)
        
        return loss

# Example usage:
# Assuming you have a model and dataset ready

# Define your model and dataset
# model = YourModel()
# train_dataset = YourTrainDataset()

# Initialize trainer
trainer = AttributeMaskingTrainer(
    model=model,
    args=TrainingArguments(output_dir="./output"),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()