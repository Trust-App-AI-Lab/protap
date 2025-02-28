import json

import torch
import numpy as np

from models.egnn.egnn import *
from data.tokenizer import ProteinTokenizer

# Example usage of ProteinTokenizer

# Initialize the tokenizer with the amino-to-index dictionary and max sequence length
tokenizer = ProteinTokenizer(max_seq_length=256, dataset='egnn-data')
tokens, masks = tokenizer.tokenize()
print(tokens[0])
print(masks[0])
print(np.array(tokens).shape)

net = EGNN_Network(
    num_tokens = 21,
    num_positions = 256,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
    dim = 32,
    depth = 3,
    num_nearest_neighbors = 8,
    coor_weights_clamp_value = 2.   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
)

feats = torch.randint(0, 21, (1, 1024)) # (1, 1024)
feats = torch.tensor(tokens[0]).reshape(1, -1)
coors = torch.randn(1, 256, 3)         # (1, 1024, 3)
mask = torch.tensor(masks[0]).reshape(1, -1).bool()   # (1, 1024)

feats_out, coors_out = net(feats, coors, mask = mask) # (1, 1024, 32), (1, 1024, 3)
print(feats_out)