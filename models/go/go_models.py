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