import transformers
from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass

from typing import Sequence, Dict

EGNN_MASK_TOKEN = 20 # The <MASK> token is used for residue prediction task.
EGNN_PAD_TOKEN = 21 # The <PAD> token is used for padding.


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
        target = torch.full_like(batch_input_ids, -100)
        
        selected_indices_list = []  # To store selected mask indices per batch.
        masked_input_ids = batch_input_ids.clone()  # Create a copy of input_ids to modify

        for i in range(batch_size):
            # Obtain the indices of the non-padding residues.
            non_pad_indices = (batch_input_ids[i] != EGNN_PAD_TOKEN).nonzero(as_tuple=True)[0]
            # According to the mask rate and the length of the aminio acids, get the number of mask which need to be masked.
            num_to_mask = max(1, int(len(non_pad_indices) * self.args.mask_ratio))
            mask_indices = torch.randperm(len(non_pad_indices))[:num_to_mask] # (num_to_mask, )
            
            selected_indices = non_pad_indices[mask_indices] # (num_to_mask, )
            target[i, selected_indices] = batch_input_ids[selected_indices]  # Assign the masked reidues with their labels.
            # Mask the selected residues with mask token.
            masked_input_ids[i, selected_indices] = EGNN_MASK_TOKEN
            
            selected_indices_list.append(selected_indices)
        
        inputs['input_ids'] = masked_input_ids  # Replace the original input_ids with the masked version
        pred = model(**inputs)  # (batch_size, seq_length, 22)
        
        # Gather logits for only the selected masked positions
        selected_indices_tensor = torch.cat(selected_indices_list)  # Flatten into a single tensor
        batch_indices = torch.arange(batch_size).repeat_interleave([len(indices) for indices in selected_indices_list])
        
        masked_logits = pred[batch_indices, selected_indices_tensor]  # (num_total_masked, 22)
        masked_targets = target[batch_indices, selected_indices_tensor]  # (num_total_masked,)

        # Compute cross-entropy loss only on masked residues
        loss = F.cross_entropy(masked_logits, masked_targets)
        
        return loss