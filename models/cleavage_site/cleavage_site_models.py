import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helper_functions import masked_mean_pooling

class EgnnCleavageModel(nn.Module):
    def __init__(self, 
                 dim,
                 egnn_model, 
                 freeze_egnn: bool=False,
                 ):
        super().__init__()
        self.egnn = egnn_model
            
        self.linear = nn.Linear(dim, 1357)
        
        if freeze_egnn:
            for param in self.egnn.parameters():
                param.requires_grad = False

    def forward(self, 
                feats, 
                coors, 
                mask, 
                site=None
            ):

        feats = self.egnn(
            feats=feats, 
            coors=coors, 
            mask=mask
        )[0]
        feats = masked_mean_pooling(feats, mask) # (batch_size, dim)

        logits = self.linear(feats)
        
        return logits
    
class Se3CleavageModel(nn.Module):
    def __init__(self, 
                 dim,
                 se3_model, 
                 freeze_se3: bool=False,
                 ):
        super().__init__()
        self.se3 = se3_model
            
        self.linear = nn.Linear(dim, 1357)
        
        if freeze_se3:
            for param in self.se3.parameters():
                param.requires_grad = False

    def forward(self, 
                feats, 
                coors, 
                mask,
                adj_mat, 
                site=None,
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

class ProteinBERTCleavageModel(nn.Module):
    def __init__(self, 
                 dim,
                 bert_model, 
                 freeze_bert: bool=False,
                 ):
        super().__init__()
        self.bert = bert_model
            
        self.linear = nn.Linear(dim, 1357)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, 
                seq,  
                mask, 
                annotation=None,
                site=None
            ):

        feats = self.bert(
            seq=seq,  
            mask=mask,
            annotation=annotation
        )[0]
        feats = masked_mean_pooling(feats, mask) # (batch_size, dim)

        logits = self.linear(feats)
        
        return logits