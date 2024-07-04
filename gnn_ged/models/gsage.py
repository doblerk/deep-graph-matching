"""
    Graph SAGE

    @References:
        - “Inductive Representation Learning on Large Graphs” paper
"""


import torch
import torch.nn.functional as F

from torch.nn import Linear
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
        
        self.layers = torch.nn.ModuleList([GraphSAGELayer(input_dim, hidden_dim) for _ in range(n_layers)])

        self.linear1 = Linear(hidden_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, n_classes)
    
    def forward(self, x, edge_index):

        # Node embeddings
        node_embeddings = []
        for layer in self.layers:
            h = layer(x, edge_index)
            # should I add this after each layer or all layers except the last one? If so, add if statement.
            h = h.relu()
            h = F.dropout(h, p=0.2, training=self.training)
            node_embeddings.append(h)
        
        # Graph-level readout
        x = global_mean_pool(node_embeddings[-1])

        # Classify
        z = self.linear1(x)
        z = z.relu()
        z = F.dropout(z, p=0.2, training=self.training)
        z = self.linear2(z)
        z = F.log_softmax(z, dim=1)

        return z