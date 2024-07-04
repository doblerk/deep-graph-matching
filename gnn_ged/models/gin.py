"""
    Graph Isomorphism Network

    @References:
        - "How Powerful are Graph Neural Networks?" paper 
        - https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sequential
from torch_geometric.nn import GINConv, global_add_pool


class GINLayer(torch.nn.Module):
    """
    A class defining the Graph Isomorphism layer

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    """
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        
        self.conv = GINConv(
            Sequential(
                Linear(input_dim,
                       hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        )
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GINModel(torch.nn.Module):
    """
    A class defining the Graph Isomorphism Network

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    n_classes: int
        number of classes
    n_layers: int
        number of hidden layers
    """
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers):
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GINLayer(input_dim, hidden_dim))
                                
        for _ in range(n_layers - 1):
            self.conv_layers.append(GINLayer(hidden_dim, hidden_dim))

        self.dense_layers = Sequential(
            Linear(hidden_dim * n_layers, hidden_dim * n_layers),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_dim * n_layers, n_classes)
        )
    
    def forward(self, x, edge_index, batch):
        
        # Node embeddings
        node_embeddings = []
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            node_embeddings.append(x)
        
        # Graph-level readout
        graph_pooled = []
        for graph in node_embeddings:
            pooled = global_add_pool(graph, batch)
            graph_pooled.append(pooled)
        
        # Concatenate graph embeddings (original version uses addition)
        h = torch.cat(graph_pooled, dim=1)

        # Classify
        z = self.dense_layers(h)
        z = F.log_softmax(z, dim=1)

        return node_embeddings[-1], z