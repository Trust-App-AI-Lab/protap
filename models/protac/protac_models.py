import os
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from datasets import load_from_disk

from data.drug_graph import sdf_to_graphs
from utils.helper_functions import masked_mean_pooling

def get_protac_graph():
    
    dataset = load_from_disk("./data/protac_1")
    # print(dataset[0])
    
    warhead = dataset['warhead']
    linker = dataset['linker']
    e3_ligand = dataset['e3_ligand']
    warhead_dirs = list(set(warhead))
    linker_dirs = list(set(linker))
    e3_ligand_dirs = list(set(e3_ligand))
    
    def extract(path):
        underscore_idx = path.rfind('_')
        dot_idx = path.rfind('.')
        return int(path[underscore_idx + 1 : dot_idx])

    warhead_graph_list, linker_graph_list, e3_ligand_list = {}, {}, {}
    for warhead_dir in warhead_dirs:
        data_dir = os.path.join('./data/protac_data/', warhead_dir)
        data_id = extract(data_dir)
        warhead_graph_list[str(data_id)] = data_dir
        
    for linker_dir in linker_dirs:
        data_dir = os.path.join('./data/protac_data/', linker_dir)
        data_id = extract(data_dir)
        linker_graph_list[str(data_id)] = data_dir
        
    for e3_ligand_dir in e3_ligand_dirs:
        data_dir = os.path.join('./data/protac_data/', e3_ligand_dir)
        data_id = extract(data_dir)
        e3_ligand_list[str(data_id)] = data_dir
    
    warhead_graphs, linker_graphs, e3_ligand_graphs = sdf_to_graphs(warhead_graph_list), sdf_to_graphs(linker_graph_list), sdf_to_graphs(e3_ligand_list)
    
    return warhead_graphs, linker_graphs, e3_ligand_graphs

class EgnnProtacModel(nn.Module):
    def __init__(self, 
                 dim,
                 poi_ligase_model,
                 warhead_ligand_model,
                 linker_model,
                 freeze_encoder: bool=False,
                 ):
        super().__init__()
        self.encoder = poi_ligase_model # Encode the poi and e3 ligase.
        self.warhead_ligand_encoder = warhead_ligand_model # Encode the warhead and e3 ligand.
        self.linker_encoder = linker_model # Encode the linker.
        
        self.linear = nn.Linear(dim, 2) # Binary classification.
        
        self.warhead_graphs, self.linker_graphs, self.e3_ligase_graphs = get_protac_graph()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, 
            poi_input_ids,
            poi_coords,
            poi_masks,
            e3_ligase_input_ids,
            e3_ligase_coords,
            e3_ligase_masks,
            warhead,
            linker,
            e3_ligand,
            label=None,
        ):

        poi_feats = self.encoder(
            feats=poi_input_ids, 
            coors=poi_coords, 
            mask=poi_masks
        )[0]
        poi_feats = masked_mean_pooling(poi_feats, poi_masks) # (batch_size, dim)
        
        e3_ligase_feats = self.encoder(
            feats=e3_ligase_input_ids,
            coors=e3_ligase_coords,
            mask=e3_ligase_masks
        )[0]
        e3_ligase_feats = masked_mean_pooling(e3_ligase_feats, e3_ligase_masks)
        
        batch_warheads, batch_linkers, batch_e3_ligands = [], [], []
        for i in range(len(poi_feats)):
            batch_warheads.append(self.warhead_graphs[str(warhead[i].item())])
            batch_linkers.append(self.linker_graphs[str(linker[i].item())])
            batch_e3_ligands.append(self.e3_ligase_graphs[str(e3_ligand[i].item())])
        
        batch_warheads, batch_linkers, batch_e3_ligands = Batch.from_data_list(batch_warheads).to(poi_feats.device), Batch.from_data_list(batch_linkers).to(poi_feats.device), Batch.from_data_list(batch_e3_ligands).to(poi_feats.device)
        
        warhead_feats = self.warhead_ligand_encoder(batch_warheads) # (batch_size, 128)
        e3_ligand_feats = self.warhead_ligand_encoder(batch_e3_ligands)
        linker_feats = self.linker_encoder(batch_linkers)

        logits = self.linear(torch.cat((poi_feats, e3_ligase_feats, warhead_feats, linker_feats, e3_ligand_feats), 1))
        
        return logits

class Se3ProtacModel(nn.Module):
    def __init__(self, 
                 dim,
                 poi_ligase_model,
                 warhead_ligand_model,
                 linker_model,
                 freeze_encoder: bool=False,
                 ):
        super().__init__()
        self.encoder = poi_ligase_model # Encode the poi and e3 ligase.
        self.warhead_ligand_encoder = warhead_ligand_model # Encode the warhead and e3 ligand.
        self.linker_encoder = linker_model # Encode the linker.
        
        self.linear = nn.Linear(dim, 2) # Binary classification.
        
        self.warhead_graphs, self.linker_graphs, self.e3_ligase_graphs = get_protac_graph()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, 
            poi_input_ids,
            poi_coords,
            poi_masks,
            e3_ligase_input_ids,
            e3_ligase_coords,
            e3_ligase_masks,
            warhead,
            linker,
            e3_ligand,
            poi_adj_mat=None,
            e3_ligase_adj_mat=None,
            label=None,
        ):

        poi_feats = self.encoder(
            feats=poi_input_ids, 
            coors=poi_coords, 
            mask=poi_masks,
            adj_mat=poi_adj_mat
        )['0']
        poi_feats = masked_mean_pooling(poi_feats, poi_masks) # (batch_size, dim)
        
        e3_ligase_feats = self.encoder(
            feats=e3_ligase_input_ids,
            coors=e3_ligase_coords,
            mask=e3_ligase_masks,
            adj_mat=e3_ligase_adj_mat
        )['0']
        e3_ligase_feats = masked_mean_pooling(e3_ligase_feats, e3_ligase_masks)
        
        batch_warheads, batch_linkers, batch_e3_ligands = [], [], []
        for i in range(len(poi_feats)):
            batch_warheads.append(self.warhead_graphs[str(warhead[i].item())])
            batch_linkers.append(self.linker_graphs[str(linker[i].item())])
            batch_e3_ligands.append(self.e3_ligase_graphs[str(e3_ligand[i].item())])
        
        batch_warheads, batch_linkers, batch_e3_ligands = Batch.from_data_list(batch_warheads).to(poi_feats.device), Batch.from_data_list(batch_linkers).to(poi_feats.device), Batch.from_data_list(batch_e3_ligands).to(poi_feats.device)
        
        warhead_feats = self.warhead_ligand_encoder(batch_warheads) # (batch_size, 128)
        e3_ligand_feats = self.warhead_ligand_encoder(batch_e3_ligands)
        linker_feats = self.linker_encoder(batch_linkers)

        logits = self.linear(torch.cat((poi_feats, e3_ligase_feats, warhead_feats, linker_feats, e3_ligand_feats), 1))
        
        return logits

class ProteinBERTProtacModel(nn.Module):
    def __init__(self, 
                 dim,
                 poi_ligase_model,
                 warhead_ligand_model,
                 linker_model,
                 freeze_encoder: bool=False,
                 ):
        super().__init__()
        self.encoder = poi_ligase_model # Encode the poi and e3 ligase.
        self.warhead_ligand_encoder = warhead_ligand_model # Encode the warhead and e3 ligand.
        self.linker_encoder = linker_model # Encode the linker.
        
        self.linear = nn.Linear(dim, 2) # Binary classification.
        
        self.warhead_graphs, self.linker_graphs, self.e3_ligase_graphs = get_protac_graph()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, 
            poi_input_ids,
            poi_masks,
            e3_ligase_input_ids,
            e3_ligase_masks,
            warhead,
            linker,
            e3_ligand,
            label=None,
            annotation=None,
        ):

        poi_feats = self.encoder(
            seq=poi_input_ids, 
            mask=poi_masks,
            annotation=annotation
        )[0]
        poi_feats = masked_mean_pooling(poi_feats, poi_masks) # (batch_size, dim)
        
        e3_ligase_feats = self.encoder(
            seq=e3_ligase_input_ids,
            mask=e3_ligase_masks,
            annotation=annotation
        )[0]
        e3_ligase_feats = masked_mean_pooling(e3_ligase_feats, e3_ligase_masks)
        
        batch_warheads, batch_linkers, batch_e3_ligands = [], [], []
        for i in range(len(poi_feats)):
            batch_warheads.append(self.warhead_graphs[str(warhead[i].item())])
            batch_linkers.append(self.linker_graphs[str(linker[i].item())])
            batch_e3_ligands.append(self.e3_ligase_graphs[str(e3_ligand[i].item())])
        
        batch_warheads, batch_linkers, batch_e3_ligands = Batch.from_data_list(batch_warheads).to(poi_feats.device), Batch.from_data_list(batch_linkers).to(poi_feats.device), Batch.from_data_list(batch_e3_ligands).to(poi_feats.device)
        
        warhead_feats = self.warhead_ligand_encoder(batch_warheads) # (batch_size, 128)
        e3_ligand_feats = self.warhead_ligand_encoder(batch_e3_ligands)
        linker_feats = self.linker_encoder(batch_linkers)

        logits = self.linear(torch.cat((poi_feats, e3_ligase_feats, warhead_feats, linker_feats, e3_ligand_feats), 1))
        
        return logits