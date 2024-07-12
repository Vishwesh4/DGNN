import sys
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from nystrom_attention import NystromAttention

from .custom_deepgcnlayer import DeepGCNLayer
from .base_class import BaseModel


@BaseModel.register("dgnn")
class DGNN_cox(BaseModel):
    '''
    Density based GNN. The model takes into consideration the different tissue types using edge weights
    to finally output hazard vector.
    Parameters:
        add_pe: Bool: Set true for adding 24 dimension randomwalk positional embedding to each patch in the WSI-bag
        edge_dim: int: Set to the dimension of edge attribute if included. Default is None, for no edge weights
    '''
    def __init__(self, num_layers=2, num_features=768, hidden_dim=128, n_classes=4, add_pe=False, edge_dim=None, **kwargs):
        super(DGNN_cox, self).__init__()
        if add_pe:
            num_features = num_features + 24 #length of pe addition
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.nystrom_norm = nn.LayerNorm(hidden_dim*self.num_layers)

        self.fc = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv = torch_geometric.nn.GATv2Conv(in_channels=hidden_dim,out_channels=hidden_dim,heads=1,concat=False, edge_dim=edge_dim)
            norm = nn.LayerNorm(hidden_dim)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=False)
            self.layers.append(layer)

        #nys4
        self.nystrom = NystromAttention(
            dim = hidden_dim*self.num_layers,
            dim_head = hidden_dim // 8,
            heads = 4,
            num_landmarks = hidden_dim * 4,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        )    

        self.hiddenlayer = nn.Sequential(*[nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        self.classifier = nn.Linear(hidden_dim, n_classes)
        print(layer)

    def forward(self, return_attn=False, **kwargs):
        data = kwargs['x_path']
        edge_index = data.edge_index
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
        else:
            edge_attr = None

        x = self.fc(data.x)
        x_ = torch.empty((len(x),0),device=x.device)
        
        for layer in self.layers:
            x, edge_attn = layer(x, edge_index, edge_attr, return_attention_weights=True)
            x_ = torch.cat([x_, x], axis=1)

        h_path, attn = self.nystrom(self.nystrom_norm(x_).unsqueeze(0),return_attn = True)
        h_path = h_path.squeeze().mean(dim=0)

        hidden_val = self.hiddenlayer(h_path)
        logits = self.classifier(hidden_val).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        if return_attn:
            return hazards, S, Y_hat, logits, None, attn, edge_attn
        else:
            return hazards, S, Y_hat, logits, None 
        