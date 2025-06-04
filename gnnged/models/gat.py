"""
    Graph Attention Network

    @References:
        - “Graph Attention Networks” paper
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sequential
from torch_geometric.nn import GATv2Conv, global_add_pool


class GATLayer(torch.nn.Module):
    """
    A class defining the Graph Attention layer

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    n_heads: int
        number of attention heads
    """
    def __init__(self, input_dim, hidden_dim, n_heads=8, dropout=0.2, concat=True) -> None:
        super().__init__()
        
        self.conv = GATv2Conv(input_dim, hidden_dim, n_heads, dropout, concat)
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Model(torch.nn.Module):
    """
    A class defining the Graph Attention Network

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    n_heads: int
        number of attention heads
    n_classes: int
        number of classes
    n_layers: int
        number of layers
    """
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers, n_heads=6) -> None:
        super().__init__()
        
        dropout = 0.5
        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GATLayer(input_dim, hidden_dim, n_heads, dropout))

        for _ in range(n_layers - 2):
            self.conv_layers.append(GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout)) # last one with just one head?
        
        self.conv_layers.append(GATLayer(hidden_dim * n_heads, hidden_dim, n_heads=1, dropout=dropout, concat=False))

        self.dense_layers = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_dim, n_classes)
        )

        self.input_proj = Linear(input_dim, hidden_dim * n_heads) if input_dim != hidden_dim * n_heads else None

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x_residual = self.input_proj(x) if self.input_proj is not None else x
        node_embeddings = []
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            x = F.elu(x)
            if i < len(self.conv_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)
            if x.shape[1] != x_residual.shape[1]:
                proj_layer = Linear(x_residual.shape[1], x.shape[1], bias=False).to(x.device)
                x_residual = proj_layer(x_residual)
            x = x + x_residual
            x_residual = x
            node_embeddings.append(x)
        
        # Graph-level readout
        x = global_add_pool(node_embeddings[-1], batch)
        
        # CLassify
        z = self.dense_layers(x)

        return node_embeddings[-1], z