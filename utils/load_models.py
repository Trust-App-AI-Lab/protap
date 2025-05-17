import torch
from collections import OrderedDict

def load_pretrain_model(
    model_path: str,
    model: torch.nn.Module
):
    state_dict = torch.load(model_path)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

def count_model_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params