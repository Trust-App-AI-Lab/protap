
import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import dgl
import math

from prot_learn.models.se3.QM9 import QM9Dataset
from prot_learn.models.se3.SE3Transformer import SE3Transformer

import warnings
warnings.filterwarnings('ignore')

class RandomRotation(object):
    def __init__(self):
        pass
    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)

config = {
    # Model Parameters
    'model': 'SE3Transformer',          # Model name
    'num_layers': 4,                     # Number of equivariant layers
    'num_degrees': 4,                    # Number of irreducible representations (Irreps) {0,1,...,num_degrees-1}
    'num_channels': 16,                  # Number of channels in intermediate layers
    'num_nlayers': 0,                    # Number of nonlinear layers
    'fully_connected': False,            # Whether to include global nodes
    'div': 4.0,                          # Ratio for low-dimensional embeddings
    'pooling': 'avg',                    # Pooling method ('avg' or 'max')
    'head': 1,                           # Number of attention heads
    # Hyperparameters
    'batch_size':64,                    # Batch size
    'lr': 1e-3,                          # Learning rate
    'num_epochs': 50,                    # Number of training epochs
    # Data
    'data_address': './data/SE3Transformer_data/QM9_data.pt',       # Data file path
    'task': 'homo',                      # QM9 task ['homo', 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    # Logging
    'name': None,                        # Run name
    'log_interval': 25,                  # Interval (in steps) for logging
    'print_interval': 250,               # Interval (in steps) for printing logs
    'save_dir': "./check_points",                # Directory to save models
    'restore': None,                     # Path to restore model
    # Others
    'num_workers': 4,                    # Number of worker processes for data loader
    'profile': False,                    # Whether to perform profiling

    # Random Seed
    'seed': 1992                         # Random seed
}


def main(config):
    if not config['name']:
        config['name'] = f"E-d{config['num_degrees']}-l{config['num_layers']}-{config['num_channels']}-{config['num_nlayers']}"

    if not os.path.isdir(config['save_dir']):
        os.makedirs(config['save_dir'])

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    config['device'] = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    print("\n\nConfiguration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("\n\n")

    train_dataset = QM9Dataset(
        config['data_address'],
        config['task'],
        mode='train',
        transform=RandomRotation()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate,
        num_workers=config['num_workers']
    )

    val_dataset = QM9Dataset(
        config['data_address'],
        config['task'],
        mode='valid'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate,
        num_workers=config['num_workers']
    )

    test_dataset = QM9Dataset(
        config['data_address'],
        config['task'],
        mode='test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate,
        num_workers=config['num_workers']
    )

    config['train_size'] = len(train_dataset)
    config['val_size'] = len(val_dataset)
    config['test_size'] = len(test_dataset)

    print(config)

    model = SE3Transformer(
        num_layers=config['num_layers'],
        atom_feature_size=train_dataset.atom_feature_size,
        num_channels=config['num_channels'],
        num_nlayers=config['num_nlayers'],
        num_degrees=config['num_degrees'],
        edge_dim=train_dataset.num_bonds,
        div=config['div'],
        pooling=config['pooling'],
        n_heads=config['head']
    )

    if config['restore'] is not None:
        model.load_state_dict(torch.load(config['restore']))
    model.to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['num_epochs'],
        eta_min=1e-4
    )

    def task_loss(pred, target, use_mean=True):
        l1_loss = torch.sum(torch.abs(pred - target))
        l2_loss = torch.sum((pred - target) ** 2)
        if use_mean:
            l1_loss /= pred.shape[0]
            l2_loss /= pred.shape[0]

        rescale_loss = train_dataset.norm2units(l1_loss, config['task'])
        return l1_loss, l2_loss, rescale_loss

    #save_path = os.path.join(config['save_dir'], config['name'] + '.pt')

    print('Begin training')
    model.train_model(task_loss, train_loader, optimizer, scheduler, config,n_epochs=1,valid_loader=val_loader)
    model.predit(task_loss, test_loader, config)


if __name__ == '__main__':
    main(config=config)


