
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from transformers import Trainer
from dataclasses import dataclass

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
    def __init__(
        self, 
        model, 
        dataset, 
        optimizer, 
        batch_size=32, 
        k=10, 
        device="cuda"
    ):
        """
        Initializes the trainer.

        Args:
            model: The EGNN model.
            dataset: The dataset containing input_ids and coordinates.
            optimizer: Optimizer for training.
            batch_size: Batch size.
            k: Number of neighbors for k-NN graph construction.
            device: Device to run the training (cuda/cpu).
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.k = k
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def create_knn_graph(self, coords):
        """
        Constructs a k-NN graph using torch_geometric.

        Args:
            coords: Tensor of shape (num_nodes, 3) representing 3D coordinates.

        Returns:
            A torch_geometric Data object containing the k-NN graph.
        """
        edge_index = knn_graph(coords, k=self.k, batch=None, loop=False)
        return Data(pos=coords, edge_index=edge_index)

    def get_positive_negative_samples(self, batch_graph):
        """
        Extracts positive and negative samples for contrastive learning.

        Args:
            batch_graph: A batch of k-NN graphs.

        Returns:
            positive_graphs: List of positive subgraphs.
            negative_graphs: List of negative samples from other graphs in the batch.
        """
        positive_graphs = []
        negative_graphs = []

        num_graphs = len(batch_graph)
        for i in range(num_graphs):
            # Extract subgraph for positive sample
            sub_nodes = batch_graph[i].edge_index[0].unique()  # Get unique nodes in the subgraph
            # TODO
            pos_edge_index, _ = subgraph(sub_nodes, batch_graph[i].edge_index, relabel_nodes=True)
            positive_graphs.append(Data(pos=batch_graph[i].pos[sub_nodes], edge_index=pos_edge_index))

            # Choose a random subgraph from another graph as negative sample
            j = (i + torch.randint(1, num_graphs, (1,)).item()) % num_graphs
            neg_nodes = batch_graph[j].edge_index[0].unique()
            neg_edge_index, _ = subgraph(neg_nodes, batch_graph[j].edge_index, relabel_nodes=True)
            negative_graphs.append(Data(pos=batch_graph[j].pos[neg_nodes], edge_index=neg_edge_index))

        return positive_graphs, negative_graphs

    def contrastive_loss(self, z, pos_z, neg_z, tau=0.1):
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
        z = F.normalize(z, dim=-1)
        pos_z = F.normalize(pos_z, dim=-1)
        neg_z = F.normalize(neg_z, dim=-1)

        # Cosine similarity
        pos_sim = torch.sum(z * pos_z, dim=-1) / tau
        neg_sim = torch.sum(z * neg_z, dim=-1) / tau

        # Contrastive loss (NT-Xent)
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss.mean()

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0

        for batch in self.dataloader:
            input_ids, coords = batch['input_ids'].to(self.device), batch['coords'].to(self.device)

            # Create graphs using k-NN
            batch_graphs = [self.create_knn_graph(coords[i]) for i in range(len(coords))]
            batch_graph = Batch.from_data_list(batch_graphs).to(self.device)

            # Extract positive and negative samples
            pos_graphs, neg_graphs = self.get_positive_negative_samples(batch_graphs)
            pos_batch = Batch.from_data_list(pos_graphs).to(self.device)
            neg_batch = Batch.from_data_list(neg_graphs).to(self.device)

            # Compute embeddings
            z = self.model(batch_graph)
            pos_z = self.model(pos_batch)
            neg_z = self.model(neg_batch)

            # Compute contrastive loss
            loss = self.contrastive_loss(z, pos_z, neg_z)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def train(self, num_epochs=10):
        """
        Runs training for a specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for.
        """
        for epoch in range(num_epochs):
            loss = self.train_one_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")