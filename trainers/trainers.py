
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from transformers import Trainer
from dataclasses import dataclass

import random
from typing import Sequence, Dict
from tqdm import tqdm

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
        # print(inputs["input_ids"])
        model.to("cuda")
        batch_input_ids, batch_coords, batch_masks = inputs['input_ids'], inputs['coords'], inputs['masks']
        batch_size = len(batch_input_ids)
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_coords = torch.stack(batch_coords, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)
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
            target[i, selected_indices] = batch_input_ids[i, selected_indices]  # Assign the masked reidues with their labels.
            # Mask the selected residues with mask token.
            masked_input_ids[i, selected_indices] = EGNN_MASK_TOKEN
            
            selected_indices_list.append(selected_indices)
        
        inputs['input_ids'] = masked_input_ids  # Replace the original input_ids with the masked version
        inputs = {
            "feats" : inputs["input_ids"].to("cuda"),
            "coors" : batch_coords.to("cuda"),
            "mask" : batch_masks.to("cuda")
        }
        
        pred = model(**inputs)[0]
        
        # Gather logits for only the selected masked positions
        selected_indices_tensor = torch.cat(selected_indices_list)  # Flatten into a single tensor
        batch_indices = torch.arange(batch_size).repeat_interleave(torch.tensor([len(indices) for indices in selected_indices_list]))
        
        masked_logits = pred[batch_indices, selected_indices_tensor].to("cuda")  # (num_total_masked, 22)
        masked_targets = target[batch_indices, selected_indices_tensor].to("cuda")  # (num_total_masked,)

        # Compute cross-entropy loss only on masked residues
        loss = F.cross_entropy(masked_logits, masked_targets)
        
        return loss

    def train(self, model, dataset, optimizer, num_epochs=10, batch_size=8, device='cuda'):
        """
        Trains the EGNN model using masked residue prediction.

        Args:
            model (torch.nn.Module): The EGNN model.
            dataset (EgnnDataset): The dataset for training.
            compute_loss (function): The loss computation function.
            optimizer (torch.optim.Optimizer): The optimizer.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            device (str): Device to use for training ('cuda' or 'cpu').
        """
    
        # Move model to device
        model.to(device)
        model.train()

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=DataCollatorForEgnnMaskResiduePrediction())

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for batch in progress_bar:
                # Move batch to device
                # batch = {key: value.to(device) for key, value in batch.items()}
                # print(batch)
                
                optimizer.zero_grad()  # Reset gradients
                
                # Compute loss
                # loss = compute_loss(model, batch)
                loss = self.compute_loss(model=model, inputs=batch)
                
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                # Track loss
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / len(dataloader):.4f}")

        print("Training complete!")


class ContrastiveEGNNTrainer(Trainer):
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
        Computes contrastive loss using NT-Xent.

        Args:
            z: Tensor of shape (batch_size, hidden_dim) - original embeddings.
            pos_z: Tensor of shape (batch_size, hidden_dim) - positive samples.
            neg_z: Tensor of shape (batch_size, hidden_dim) - negative samples.
            tau: Temperature parameter for contrastive loss.

        Returns:
            Contrastive loss scalar.
        """
        model.to("cuda")
        batch_input_ids, batch_coords, batch_masks = inputs['input_ids'], inputs['coords'], inputs['masks']
        batch_size = len(batch_input_ids)
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_coords = torch.stack(batch_coords, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)

        embeddings = []  # Store embeddings for all views
        SAMPLING_METHODS = {
            "subspace": self.get_subspace,
            "subsequence": self.get_subsequence,
        }
        
        # Store the two views for each protein
        sub_input_ids_x, sub_coords_x, sub_masks_x = [], [], []
        sub_input_ids_y, sub_coords_y, sub_masks_y = [], [], []

        for i in range(batch_size):

            # Randomly select two different sampling methods
            method_x, method_y = random.sample(list(SAMPLING_METHODS.keys()), 2)

            # Generate two subgraphs
            view_x = SAMPLING_METHODS[method_x](batch_input_ids[i], batch_coords[i], batch_masks[i])
            view_y = SAMPLING_METHODS[method_y](batch_input_ids[i], batch_coords[i], batch_masks[i])

            sub_input_ids_x.append(view_x[0])
            sub_coords_x.append(view_x[1])
            sub_masks_x.append(view_x[2])

            sub_input_ids_y.append(view_y[0])
            sub_coords_y.append(view_y[1])
            sub_masks_y.append(view_y[2])

        # Convert lists to batch tensors
        sub_input_ids_x = torch.stack(sub_input_ids_x)  # (batch_size, max_nodes)
        sub_coords_x = torch.stack(sub_coords_x)  # (batch_size, max_nodes, 3)
        sub_masks_x = torch.stack(sub_masks_x)  # (batch_size, max_nodes)

        sub_input_ids_y = torch.stack(sub_input_ids_y)
        sub_coords_y = torch.stack(sub_coords_y)
        sub_masks_y = torch.stack(sub_masks_y)

        inputs_x = {
            "feats" : sub_input_ids_x.to("cuda"),
            "coors" : sub_coords_x.to("cuda"),
            "mask" : sub_masks_x.to("cuda")
        }
        inputs_y = {
            "feats" : sub_input_ids_y.to("cuda"),
            "coors" : sub_coords_y.to("cuda"),
            "mask" : sub_masks_y.to("cuda")
        }
        
        # Compute embeddings using the model
        emb_1 = model(inputs_x)  # (batch_size, max_nodes, embedding_dim)
        emb_2 = model(inputs_y)  # (batch_size, max_nodes, embedding_dim)
        
        def masked_mean_pooling(embeddings, masks):
            """
            Perform mean pooling over node embeddings while considering valid nodes.

            Args:
            - embeddings: (batch_size, max_nodes, embedding_dim), node embeddings.
            - masks: (batch_size, max_nodes), binary mask indicating valid nodes.

            Returns:
            - graph_embeddings: (batch_size, embedding_dim), pooled graph representation.
            """
            masks = masks.float().unsqueeze(-1)  # (batch_size, max_nodes, 1) for broadcasting
            sum_embeddings = torch.sum(embeddings * masks, dim=1)  # Sum over valid nodes
            valid_node_count = torch.clamp(masks.sum(dim=1), min=1)  # Avoid division by zero
            
            return sum_embeddings / valid_node_count  # Compute mean while ignoring padding

        # Apply masked mean pooling
        graph_emb_x = masked_mean_pooling(emb_1, sub_masks_x)  # (batch_size, embedding_dim)
        graph_emb_y = masked_mean_pooling(emb_2, sub_masks_y)  # (batch_size, embedding_dim)

        # Normalize embeddings
        graph_emb_x = F.normalize(graph_emb_x, dim=1)
        graph_emb_y = F.normalize(graph_emb_y, dim=1)

        # Concatenate both embeddings for similarity calculation
        embeddings = torch.cat([graph_emb_x, graph_emb_y], dim=0)  # (2 * batch_size, embedding_dim)

        # Compute cosine similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.T)  # (2 * batch_size, 2 * batch_size)
        sim_matrix = sim_matrix / self.args.temperature  # Apply temperature scaling

        # Create labels: Each sample's positive pair is its adjacent index
        labels = torch.arange(0, 2 * batch_size, 2)  # (batch_size,)
        labels = torch.cat([labels + 1, labels], dim=0)  # (2 * batch_size,) - Each sample's positive pair

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def get_subsequence(
        self,
        input_ids,
        coords,
        masks,
    ):  
        """
        Function for constructing a subgraph g(x) of a given batch data. 
        Here, the subgraph is a random subsequence of the input amnio acids sequence.
        """
        seq_length = input_ids.shape[0]

        # Find the first padding token index
        pad_indices = (input_ids == EGNN_PAD_TOKEN).nonzero(as_tuple=True)[0]
        first_pad_idx = pad_indices[0].item() if len(pad_indices) > 0 else seq_length  # If no padding, use full length

        # Ensure valid sampling range
        max_l = max(0, first_pad_idx - self.args.subseq_length)  # The last valid start index
        if max_l > 0:
            l = torch.randint(0, max_l + 1, (1,)).item()  # Sample a start index within valid range
        else:
            l = 0  # If sequence is too short, take from the beginning

        r = l + self.args.subseq_length

        sub_input_ids = input_ids[l:r]
        sub_coords = coords[l:r]
        sub_masks = masks[l:r]

        return sub_input_ids, sub_coords, sub_masks
    
    def get_subspace(
        self,
        input_ids,
        coords,
        masks,
    ):
        """
        Function for constructing a subgraph g(y) of a given batch data. 
        We randomly sample a residue p as the center and select all residues within a Euclidean ball with a predefined radius d.
        """
        seq_length = input_ids.shape[0]

        # Get valid indices (non-padding residues)
        valid_indices = (input_ids != EGNN_PAD_TOKEN).nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid amino acids found in the input sequence.")

        # Randomly select a center residue from valid indices
        center_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()

        # Compute Euclidean distances only for valid (non-padding) residues
        valid_coords = coords[valid_indices]  # Only non-padding coordinates
        center_coord = coords[center_idx].unsqueeze(0)  # Shape: (1, 3)

        distances = torch.norm(valid_coords - center_coord, dim=1)  # Compute Euclidean distance

        # Select valid residues within radius d
        selected_valid_indices = valid_indices[(distances <= self.args.d).nonzero(as_tuple=True)[0]]

        # Ensure at least one node is selected (fallback to center if no others found)
        if len(selected_valid_indices) == 0:
            selected_valid_indices = torch.tensor([center_idx], dtype=torch.long)

        # If too many nodes are selected, randomly downsample
        if len(selected_valid_indices) > self.args.max_nodes:
            selected_valid_indices = selected_valid_indices[torch.randperm(len(selected_valid_indices))[:self.args.max_nodes]]

        # Extract the subgraph
        sub_input_ids = input_ids[selected_valid_indices]
        sub_coords = coords[selected_valid_indices]
        sub_masks = masks[selected_valid_indices]

        # Padding to `max_nodes`
        pad_length = self.args.max_nodes - len(selected_valid_indices)
        if pad_length > 0:
            pad_input_ids = torch.full((pad_length,), EGNN_PAD_TOKEN)
            pad_coords = torch.zeros((pad_length, 3))
            pad_masks = torch.zeros((pad_length,), dtype=torch.bool)

            sub_input_ids = torch.cat([sub_input_ids, pad_input_ids], dim=0)
            sub_coords = torch.cat([sub_coords, pad_coords], dim=0)
            sub_masks = torch.cat([sub_masks, pad_masks], dim=0)

        return sub_input_ids, sub_coords, sub_masks

    def train(self, model, dataset, optimizer, num_epochs=10, batch_size=8, device='cuda'):
        """
        Runs training for a specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for.
        """
        # Move model to device
        model.to(device)
        model.train()

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=DataCollatorForEgnnMaskResiduePrediction())

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for batch in progress_bar:
                # Move batch to device
                # batch = {key: value.to(device) for key, value in batch.items()}
                # print(batch)
                
                optimizer.zero_grad()  # Reset gradients
                
                # Compute loss
                # loss = compute_loss(model, batch)
                loss = self.compute_loss(model=model, inputs=batch)
                
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                # Track loss
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / len(dataloader):.4f}")

        print("Training complete!")