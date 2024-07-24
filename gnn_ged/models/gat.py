"""
    Graph Attention Network

    @References:
        - “Graph Attention Networks” paper
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, ReLU, Dropout, Sequential
from torch_geometric.nn import GATv2Conv, global_mean_pool


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
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers, n_heads=8) -> None:
        super().__init__()
        
        dropout = 0.5
        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GATLayer(input_dim, hidden_dim, n_heads, dropout))

        for _ in range(n_layers - 2):
            self.conv_layers.append(GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout)) # last one with just one head?
        
        self.conv_layers.append(GATLayer(hidden_dim * n_heads, hidden_dim, n_heads=1, dropout=dropout, concat=False))

        self.dense_layers = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_dim, n_classes)
        )

    def forward(self, x, edge_index, batch):
        # Node embeddings
        node_embeddings = []
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)
                x = F.elu(x)
            node_embeddings.append(x)
        
        # Graph-level readout
        x = global_mean_pool(node_embeddings[-1], batch)
        
        # CLassify
        z = self.dense_layers(x)
        z = F.log_softmax(z, dim=1)

        return node_embeddings[-1], z