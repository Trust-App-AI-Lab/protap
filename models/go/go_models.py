import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helper_functions import masked_mean_pooling

class EgnnGOModel(nn.Module):
    def __init__(self, 
                 dim,
                 egnn_model, 
                 go_term=None,
                 freeze_egnn: bool=False,
                 ):
        super().__init__()
        self.egnn = egnn_model
        
        if go_term == 'biological_process':
            self.out_dim = 1943
        elif go_term == 'molecular_function':
            self.out_dim = 489
        elif go_term == 'cellular_component':
            self.out_dim = 320
            
        self.linear = nn.Linear(dim, self.out_dim)
        
        if freeze_egnn:
            for param in self.egnn.parameters():
                param.requires_grad = False

    def forward(self, 
                feats, 
                coors, 
                mask, 
                go=None,
            ):

        feats = self.egnn(
            feats=feats, 
            coors=coors, 
            mask=mask
        )[0]
        feats = masked_mean_pooling(feats, mask) # (batch_size, dim)

        logits = self.linear(feats)
        
        return logits
    
class Se3GOModel(nn.Module):
    def __init__(self, 
                 dim,
                 se3_model, 
                 go_term=None,
                 freeze_se3: bool=False,
                 ):
        super().__init__()
        self.se3 = se3_model
        
        if go_term == 'biological_process':
            self.out_dim = 1943
        elif go_term == 'molecular_function':
            self.out_dim = 489
        elif go_term == 'cellular_component':
            self.out_dim = 320
            
        self.linear = nn.Linear(dim, self.out_dim)
        
        if freeze_se3:
            for param in self.se3.parameters():
                param.requires_grad = False

    def forward(self, 
                feats, 
                coors, 
                mask,
                adj_mat, 
                go=None,
            ):

        feats = self.se3(
            feats=feats, 
            coors=coors, 
            mask=mask,
            adj_mat=adj_mat
        )['0']
        feats = masked_mean_pooling(feats, mask) # (batch_size, dim)

        logits = self.linear(feats)
        
        return logits

class ProteinBERTGOModel(nn.Module):
    def __init__(self, 
                 dim,
                 bert_model, 
                 go_term=None,
                 freeze_bert: bool=False,
                 ):
        super().__init__()
        self.bert = bert_model
        
        if go_term == 'biological_process':
            self.out_dim = 1943
        elif go_term == 'molecular_function':
            self.out_dim = 489
        elif go_term == 'cellular_component':
            self.out_dim = 320
            
        self.linear = nn.Linear(dim, self.out_dim)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, 
                seq,  
                mask, 
                go=None,
                annotation=None
            ):

        feats = self.bert(
            seq=seq,  
            mask=mask,
            annotation=annotation
        )[0]
        feats = masked_mean_pooling(feats, mask) # (batch_size, dim)

        logits = self.linear(feats)
        
        return logits