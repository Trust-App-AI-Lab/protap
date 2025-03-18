import torch
from collections import OrderedDict

def load_egnn_model(
    model_path: str,
    model: torch.nn.Module
):
    state_dict = torch.load(model_path)
    # device = torch.device("cuda")
    # net.load_state_dict(torch.load('egnn_node.pt', weights_only=True, map_location="cpu"))
    # # net.to(device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    
    return model