import json

import torch
import numpy as np

from prot_learn.models.egnn.egnn import *

# load the protein data.
with open('/home/yuliangyan/Code/gvp-pytorch/data/ts50.json', 'r') as json_file:
    data = json.load(json_file)
# raw data shape: (50, )
# keys: "name", "seq", "coords"
sequences = [protein['seq'] for protein in data]
# max protein sequence length: 173
max_length = np.max([len(seq) for seq in sequences])
max_index = np.argmax([len(seq) for seq in sequences])
amino_dict = list(set(sequences[max_index]))
amino_dict.append('<PAD>')
amino2dict = {amino : idx for idx, amino in enumerate(amino_dict)}
print(amino_dict)
print(amino2dict)

net = EGNN_Network(
    num_tokens = 21,
    num_positions = 1024,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
    dim = 32,
    depth = 3,
    num_nearest_neighbors = 8,
    coor_weights_clamp_value = 2.   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
)

feats = torch.randint(0, 21, (1, 1024)) # (1, 1024)
coors = torch.randn(1, 1024, 3)         # (1, 1024, 3)
mask = torch.ones_like(feats).bool()    # (1, 1024)

feats_out, coors_out = net(feats, coors, mask = mask) # (1, 1024, 32), (1, 1024, 3)
print(feats_out)