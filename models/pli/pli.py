import torch
import torch.nn as nn
import torch.nn.functional as F

from drug_gvp.drug_gvp import DrugGVPModel
from egnn.egnn import EGNN_Network

class EgnnPLIModel(nn.Module):
    def __init__(self, 
                 egnn_model, 
                 drug_model,
                 freeze_egnn: bool=False,
                 ):
        super().__init__()
        self.egnn = egnn_model
        self.drug_model = drug_model
        
        # self.drug_structure =  # TODO
        
        if freeze_egnn:
            for param in self.egnn.parameters():
                param.requires_grad = False

    def forward(self, 
                input_ids, 
                coords, 
                mask, 
                family_labels
                ):

        feats, family_emb = self.egnn(
            feats=input_ids, coors=coords, mask=mask, family_labels=family_labels
        )
        graph_repr = masked_mean_pooling(feats, mask)  # (batch_size, dim)
        graph_repr = F.normalize(graph_repr, dim=-1)

        drug_output = self.drug_model(graph_repr)

        return {
            "egnn_feat": graph_repr,
            "family_emb": family_emb,
            "drug_output": drug_output
        }