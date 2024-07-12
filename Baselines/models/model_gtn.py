'''
Code modified from https://github.com/vkola-lab/tmi2022
'''
import torch
import torch_geometric
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import dense_mincut_pool

from ..model_utils.ViT import *
from typing import Optional

class DeepGCNLayer(torch.nn.Module):
    r"""The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).

    * **Res+** (:obj:`"res+"`):

    .. math::
        \text{Normalization}\to\text{Activation}\to\text{Dropout}\to
        \text{GraphConv}\to\text{Res}

    * **Res** (:obj:`"res"`) / **Dense** (:obj:`"dense"`) / **Plain**
      (:obj:`"plain"`):

    .. math::
        \text{GraphConv}\to\text{Normalization}\to\text{Activation}\to
        \text{Res/Dense/Plain}\to\text{Dropout}

    .. note::

        For an example of using :obj:`GENConv`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        conv (torch.nn.Module, optional): the GCN operator.
            (default: :obj:`None`)
        norm (torch.nn.Module): the normalization layer. (default: :obj:`None`)
        act (torch.nn.Module): the activation layer. (default: :obj:`None`)
        block (str, optional): The skip connection operation to use
            (:obj:`"res+"`, :obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`).
            (default: :obj:`"res+"`)
        dropout (float, optional): Whether to apply or dropout.
            (default: :obj:`0.`)
        ckpt_grad (bool, optional): If set to :obj:`True`, will checkpoint this
            part of the model. Checkpointing works by trading compute for
            memory, since intermediate activations do not need to be kept in
            memory. Set this to :obj:`True` in case you encounter out-of-memory
            errors while going deep. (default: :obj:`False`)
    
            Modified the original code to return attention weights for visualization purposes
    """
    def __init__(
        self,
        conv: Optional[Module] = None,
        norm: Optional[Module] = None,
        act: Optional[Module] = None,
        block: str = 'res+',
        dropout: float = 0.,
        ckpt_grad: bool = False,
    ):
        super().__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, return_attention_weights=False, **kwargs) -> Tensor:
        """"""
        args = list(args)
        x = args.pop(0)
        # x = kwargs.pop("x")

        if self.block == 'res+':
            h = x
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.conv, h, *args, **kwargs)
            else:
                #Valid for inference
                if return_attention_weights:
                    h, attn_weights = self.conv(h, *args, return_attention_weights=return_attention_weights, **kwargs)
                else:
                    h = self.conv(h, *args, **kwargs)

            if return_attention_weights:
                return x + h, attn_weights
            else:
                return x + h

        else:
            if self.conv is not None and self.ckpt_grad and x.requires_grad:
                h = checkpoint(self.conv, x, *args, **kwargs)
            else:
                #Valid for inference
                if return_attention_weights:
                    h, attn_weights = self.conv(x, *args, return_attention_weights=return_attention_weights, **kwargs)
                else:
                    h = self.conv(x, *args, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == 'res':
                h = x + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass

            if return_attention_weights:
                return F.dropout(h, p=self.dropout, training=self.training), attn_weights
            else:
                return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block}, conv={self.conv}, norm={self.norm}, activation={self.act}, dropout={self.dropout})'

class GTN_cox_original(torch.nn.Module):
    def __init__(self, num_layers=4, resample=0, num_features=768, hidden_dim=128, pool=False, dropout=0.25, n_classes=4):
        super(GTN_cox_original, self).__init__()
        self.pool = pool
        self.num_layers = num_layers
        self.resample = resample
        self.node_cluster_num = 200

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv = torch_geometric.nn.GCNConv(hidden_dim,hidden_dim)
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=False)
            self.layers.append(layer)

        self.pool_layer = nn.Linear(hidden_dim, self.node_cluster_num)
        self.transformer = VisionTransformer(num_classes=n_classes, embed_dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self,  **kwargs):
        data = kwargs['x_path']
        adj = torch_geometric.utils.to_dense_adj(data.edge_index)
        edge_index = data.edge_index

        x = self.fc(data.x)
        x_ = x 

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        h_path = x_
        s = self.pool_layer(h_path)
        X, adj, mc1, o1 = dense_mincut_pool(h_path, adj, s)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        logits  = self.transformer(X) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, None, None