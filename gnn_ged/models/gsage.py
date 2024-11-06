"""
    Graph SAGE

    @References:
        - “Inductive Representation Learning on Large Graphs” paper
"""


import torch
import torch.nn.functional as F

from torch.nn import Linear, ReLU, Dropout, Sequential
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGELayer(torch.nn.Module):
    """
    A class defining the GraphSAGE layer

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    """
    def __init__(self, input_dim, hidden_dim, aggr='mean') -> None:
        super().__init__()
        
        self.conv = SAGEConv(input_dim, hidden_dim, aggr=aggr)
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Model(torch.nn.Module):
    """
    A class defining the GraphSAGE Network

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    n_classes: int
        number of classes
    """
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers) -> None:
        super().__init__()
        
        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GraphSAGELayer(input_dim, hidden_dim))

        for _ in range(n_layers - 1):
            self.conv_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))

        self.dense_layers = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_dim, n_classes)
        )

        self.input_proj = Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
    
    def forward(self, x, edge_index, batch):

        # Node embeddings
        x_residual = self.input_proj(x) if self.input_proj is not None else x
        node_embeddings = []
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)   
            if i < len(self.conv_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)
            x = x + x_residual
            x_residual = x
            node_embeddings.append(x)
        
        # Graph-level readout
        x = global_mean_pool(node_embeddings[-1], batch)

        # Classify
        z = self.dense_layers(x)

        return node_embeddings[-1], z