import os
import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATv2Conv, GraphNorm, global_add_pool, global_mean_pool


class GCNLayer(torch.nn.Module):
    """
    A class defining the Graph Convolutional layer

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    """
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        
        self.conv = GCNConv(input_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GCN(torch.nn.Module):
    """
    A class defining the Graph Convolutional Network

    Attributes
    ----------
    n_features: int
        number of features per node
    n_channels: int
        number of channels
    n_classes: int
        number of classes
    n_layers: int
        number of hidden layers
    """
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList([GINLayer(input_dim, hidden_dim) for _ in range(n_layers)])

        self.norms = torch.nn.ModuleList([GraphNorm(hidden_dim) for _ in range(n_layers)])

        self.linear1 = Linear(hidden_dim * n_layers, hidden_dim * n_layers)
        self.linear2 = Linear(hidden_dim * n_layers, n_classes)
    
    def forward(self, x, edge_index, batch):
        
        # Node embeddings
        node_embeddings = []
        for i in range(self.layers):
            h = self.layers[i](x, edge_index)
            h = self.norms[i](x, batch)
            if i < len(self.layers):
                h = h.relu()
            node_embeddings.append(h)

        # Graph-level readout
        x = global_mean_pool(node_embeddings[-1])

        # Classify
        z = self.linear1(x)
        z = z.relu()
        z = F.dropout(z, p=0.2, training=self.training)
        z = self.linear2(z)
        z = F.log_softmax(z, dim=1)

        return node_embeddings[-1], z


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

        self.layers = torch.nn.ModuleList([GINLayer(input_dim, hidden_dim) for _ in range(n_layers)])

        self.linear1 = Linear(hidden_dim * n_layers, hidden_dim * n_layers)
        self.linear2 = Linear(hidden_dim * n_layers, n_classes)
    
    def forward(self, x, edge_index, batch):
        
        # Node embeddings
        node_embeddings = []
        for layer in self.layers:
            node_embeddings.append(layer(x, edge_index))

        # Graph-level readout
        graph_pooled = []
        for graph in node_embeddings:
            graph_pooled.append(global_add_pool(graph, batch))
        
        # Concatenate graph embeddings (original version uses addition)
        h = torch.cat(graph_pooled, dim=1)

        # Classify
        z = self.linear1(h)
        z = z.relu()
        z = F.dropout(z, p=0.2, training=self.training)
        z = self.linear2(z)
        z = F.log_softmax(z, dim=1)

        return node_embeddings[-1], z


class GCNBlock(torch.nn.Module):
    """
    A class defining the Graph Convolutional Network using concatenated skip connections

    Attributes
    ----------
    n_features: int
        number of features per node
    n_channels: int
        number of channels
    """
    def __init__(self, n_features, n_channels):
        super().__init__()
        self._block = Sequential(
            'x, edge_index, batch', [
                (BatchNorm1d(n_channels), 'x, edge_index -> x'),
                (GCNConv(in_channels=n_features,
                        out_channels=n_channels), 'x, edge_index -> x'),
                (ReLU(inplace=True), 'x, edge_index -> x'),
            ]
        )

    def forward(self, x, edge_index):
        out = self._block(x, edge_index)
        out = torch.cat([out, x], dim=1)
        return out


class GCNBlockStack(torch.nn.Module):
    """
    A class defining the stack of multiple GCN blocks

    Attributes
    ----------
    n_features: int
        number of features per node
    n_channels: int
        number of channels
    n_classes: int
        number of classes
    n_layers: int
        number of layers
    """
    def __init__(self, n_features, n_channels, n_classes, n_layers):
        super().__init__()
        self._n_layers = n_layers

        self._layers = []
        for layer_idx in range(0, self._n_layers):
            if layer_idx < 1:
                n_features_in = n_features
                n_features_out = n_features + n_channels
                # self._layers.append(GCNBlock(n_features_in, n_features_out))
                self._layers.extend(self.GNNBlock(n_features, n_features_in, n_features_out))
            else:
                n_features_in = n_features_out
                n_features_out = n_features_in + n_channels
                # self._layers.append(GCNBlock(n_features_in, n_features_out))
                self._layers.extend(self.GNNBlock(n_features_in, n_features_in, n_features_out))

        self._stacked_layers = Sequential('x, edge_index, batch', self._layers)

        self._linear1 = [
            (Linear(
                in_features=n_features_out,
                out_features=n_features_out//4,
            )),
            'x -> x'
        ]

        self._linear2 = [
            (Linear(
                in_features=n_features_out//4,
                out_features=n_classes,
            )),
            'x -> x'
        ]

        self._linear_layers = Sequential('x', [self._linear1, self._linear2])
        # self._layers.extend([self._linear1, self._linear2])
    
    def GNNBlock(self, n_channels, n_features_in, n_features_out):
        _block = [
            (BatchNorm1d(n_channels), 'x -> x'),
            (GCNConv(in_channels=n_features_in, out_channels=n_features_out), 'x, edge_index -> x'),
            (ReLU(inplace=True), 'x -> x'),
        ]
        return _block

    def forward(self, x, edge_index, batch):
        # for i in range(self._n_layers):
        #     x = self._stacked_layers[i](x, edge_index)
        # p = global_mean_pool(x, batch)
        # z = self._linear1(p)
        # z = self._linear2(z)
        x = self._stacked_layers(x, edge_index, batch)
        p = global_mean_pool(x, batch)
        out = self._linear_layers(p)
        # out = self._stacked_layers(x, edge_index, batch)
        return out 


class GNNConcatSkipConnections(torch.nn.Module):
    """
    TODO
    """
    def __init__(self):
        super(GNNConcatSkipConnections, self).__init__()

        self._model = Sequential(
            'x, edge_index, batch',
            [
                (BatchNorm1d(7), 'x, edge_index -> x'),
                (GCNConv(in_channels=7, out_channels=71), 'x, edge_index -> x'),
                (ReLU(inplace=True), 'x, edge_index -> x'),
                (BatchNorm1d(71), 'x, edge_index -> x'),
                (GCNConv(in_channels=71, out_channels=135), 'x, edge_index -> x'),
                (ReLU(inplace=True), 'x, edge_index -> x'),
                (BatchNorm1d(135), 'x, edge_index -> x'),
                (GCNConv(in_channels=135, out_channels=199), 'x, edge_index -> x'),
                (ReLU(inplace=True), 'x, edge_index -> x'),
                (global_mean_pool, 'x, batch -> x'),
                (Linear(199, 199//4), 'x -> x'),
                (Linear(199//4, 2), 'x -> x'),
            ]
        )
    
    def forward(self, x, edge_index, batch):
        out = self._model(x, edge_index, batch)
        return out


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


class GraphSAGE(torch.nn.Module):
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


class GATLayer(torch.nn.Module):
    """
    A class defining the Graph Attention layer

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    """
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        
        self.conv = GATv2Conv(input_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GAT(torch.nn.module):
    """
    A class defining the Graph Attention Network

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
        
        self.layers = torch.nn.ModuleList([GATLayer(input_dim, hidden_dim) for _ in range(n_layers)])

        self.linear1 = Linear(hidden_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, n_classes)
    
    def forward(self, x, edge_index):

        # Node embeddings
        node_embeddings = []
        for layer in self.layers:
            h = layer(x, edge_index)
            h = h.elu()
            h = F.dropout(h, p=0.2, training=self.training)
            node_embeddings.append(h)
        
        # Graph-level readout
        x = global_mean_pool(node_embeddings[-1])

        # CLassify
        z = self.linear1(x)
        z = z.relu()
        z = F.dropout(z, p=0.2, training=self.training)
        z = self.linear2(z)
        z = F.log_softmax(z, dim=1)

        return z