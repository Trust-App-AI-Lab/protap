

import os
import sys
import warnings
from tqdm import tqdm

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import dgl
import math

from prot_learn.models.se3.equivariant_attention.modules import (
    GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
)
from prot_learn.models.se3.equivariant_attention.fibers import Fiber



class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, div: float = 4, pooling: str = 'avg', n_heads: int = 1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {
            'in': Fiber(1, atom_feature_size),
            'mid': Fiber(num_degrees, self.num_channels),
            'out': Fiber(1, num_degrees * self.num_channels)
        }

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                   div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # Encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h

    def train_model(self, loss_fnc, train_loader, optimizer, scheduler, config, n_epochs, valid_loader=None,checkpoint_path='check_points/se3.pt'):
        """
        训练 SE3Transformer 模型。

        参数：
        - loss_fnc: 损失函数
        - train_loader: 训练数据加载器
        - optimizer: 优化器
        - scheduler: 学习率调度器
        - config: 配置字典
        - n_epochs: 训练轮数
        - valid_loader: 验证数据加载器（可选）
        """
        device = config['device']
        self.to(device)

        for epoch in range(1, n_epochs + 1):
            self.train()
            total_train_loss = 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} - Training", leave=False)

            for i, (g, y) in enumerate(train_bar):
                g = g.to(device)
                y = y.to(device).view(-1, 1)  # 确保 y 的形状为 [batch_size, 1]

                optimizer.zero_grad()
                pred = self(g)
                l1_loss, __, rescale_loss = loss_fnc(pred, y)

                l1_loss.backward()
                optimizer.step()

                total_train_loss += l1_loss.item()

                train_bar.set_postfix({'L1 Loss': f"{l1_loss.item():.5f}",
                                           'Rescale Loss': f"{rescale_loss.item():.5f}"})

                scheduler.step(epoch + i / len(train_loader))

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch}/{n_epochs}, Training Loss: {avg_train_loss:.4f}")

        if valid_loader is not None:
            self.eval()
            rloss = 0
            with torch.no_grad():
                for i, (g, y) in enumerate(valid_loader):
                    g = g.to(config['device'])
                    y = y.to(config['device'])

                    pred = self(g).detach()
                    __, __, rl = loss_fnc(pred, y, use_mean=False)
                    rloss += rl
            rloss /= config['val_size']
            print(f"val rescale loss: {rloss:.5f} [units]")

        current_directory = os.getcwd()
        save_pt = os.path.join(current_directory, checkpoint_path)
        torch.save(self.state_dict(), save_pt)
        print(f"Model saved to {save_pt}")


    def predit(self,loss_fnc, dataloader, config):
        ##### this function don't work now, still need to implement
        self.eval()
        rloss = 0
        with torch.no_grad():
            for i, (g, y) in enumerate(dataloader):
                g = g.to(config['device'])
                y = y.to(config['device'])
                # 运行模型前向并计算损失
                pred = self(g).detach()
                __, __, rl = loss_fnc(pred, y, use_mean=False)
                rloss += rl
        rloss /= config['test_size']
        print(f"rescale loss: {rloss:.5f} [units]")
        # return output


