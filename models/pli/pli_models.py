import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk

from data.drug_graph import sdf_to_graphs
from utils.helper_functions import masked_mean_pooling

def get_drug_graph():
    
    dataset = load_from_disk("protein_drug_1") # {input_ids, coords, masks, drugs, y}
    # print(dataset[0])
    
    drug_dir = '/mnt/data/protein_data/pli_data/davis_mol3d_sdf'
    drug = dataset['drug']
    drug_ids = list(set(drug))
    
    graph_list = {}
    for drug_id in drug_ids:
        data_dir = os.path.join(drug_dir, str(drug_id) + '.sdf')
        graph_list[drug_id] = data_dir
    
    drug_graphs = sdf_to_graphs(graph_list)
    
    return drug_graphs

class EgnnPLIModel(nn.Module):
    def __init__(self, 
                 dim,
                 egnn_model, 
                 drug_model,
                 freeze_egnn: bool=False,
                 ):
        super().__init__()
        self.egnn = egnn_model
        self.drug_model = drug_model
        
        self.linear = nn.Linear(dim, 1)
        
        self.drug_graphs = get_drug_graph()
        
        if freeze_egnn:
            for param in self.egnn.parameters():
                param.requires_grad = False

    def forward(self, 
                feats, 
                coors, 
                mask, 
                drugs,
            ):

        feats = self.egnn(
            feats=feats, 
            coors=coors, 
            mask=mask
        )
        feats = masked_mean_pooling(feats, mask) # (batch_size, dim)

        batch_drugs = []
        for i in len(feats):
            batch_drugs.append(self.drug_graphs[drugs[i].item()])

        batch_drugs.to(feats.device)
        drug_output = self.drug_model(batch_drugs) # (batch_size, dim)

        preds = self.linear(torch.cat((feats, drug_output), 1))
        
        return preds