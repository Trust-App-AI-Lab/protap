import json

import torch
import numpy as np

from models.egnn.egnn import *
from data.dataset import EgnnDataset
from data.tokenizers import ProteinTokenizer


tokenizer = ProteinTokenizer(max_seq_length=256, dataset='egnn-data')
dataset = EgnnDataset(tokenizer=tokenizer)
print(dataset[0])

net = EGNN_Network(
    num_tokens=21,
    num_positions=256,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
    dim=2,
    depth=3,
    num_nearest_neighbors=8,
    coor_weights_clamp_value=2.   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
)

feats = dataset[0:1]['input_ids']
coords = dataset[0:1]['coords']
masks = dataset[0:1]['masks']

feats_out, coors_out = net(feats, coords, mask = masks) # (1, 1024, 32), (1, 1024, 3)
print(feats_out.shape)
print(feats_out)