"""
    Graph Convolutional Network

    @References:
        - “Semi-supervised Classification with Graph Convolutional Networks” paper
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, ReLU, Dropout, Sequential
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool


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


class Model(torch.nn.Module):
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
        
        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GCNLayer(input_dim, hidden_dim))

        for _ in range(n_layers - 1):
            self.conv_layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = torch.nn.ModuleList(
            [GraphNorm(hidden_dim) for _ in range(n_layers)]
        )

        self.dense_layers = Sequential(
            Linear(hidden_dim * n_layers, hidden_dim * n_layers),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_dim * n_layers, n_classes)
        )
    
    def forward(self, x, edge_index, batch):
        
        # Node embeddings
        node_embeddings = []
        for i in range(self.layers):
            h = self.conv_layers[i](x, edge_index)
            h = self.norms[i](x, batch)
            if i < len(self.layers):
                h = h.relu()
            node_embeddings.append(h)

        # Graph-level readout
        x = global_mean_pool(node_embeddings[-1])

        # Classify
        z = self.dense_layers(x)
        z = F.log_softmax(z, dim=1)

        return node_embeddings[-1], z


# class GCNBlock(torch.nn.Module):
#     """
#     A class defining the Graph Convolutional Network using concatenated skip connections

#     Attributes
#     ----------
#     n_features: int
#         number of features per node
#     n_channels: int
#         number of channels
#     """
#     def __init__(self, n_features, n_channels):
#         super().__init__()
#         self._block = Sequential(
#             'x, edge_index, batch', [
#                 (BatchNorm1d(n_channels), 'x, edge_index -> x'),
#                 (GCNConv(in_channels=n_features,
#                         out_channels=n_channels), 'x, edge_index -> x'),
#                 (ReLU(inplace=True), 'x, edge_index -> x'),
#             ]
#         )

#     def forward(self, x, edge_index):
#         out = self._block(x, edge_index)
#         out = torch.cat([out, x], dim=1)
#         return out


# class GCNBlockStack(torch.nn.Module):
#     """
#     A class defining the stack of multiple GCN blocks

#     Attributes
#     ----------
#     n_features: int
#         number of features per node
#     n_channels: int
#         number of channels
#     n_classes: int
#         number of classes
#     n_layers: int
#         number of layers
#     """
#     def __init__(self, n_features, n_channels, n_classes, n_layers):
#         super().__init__()
#         self._n_layers = n_layers

#         self._layers = []
#         for layer_idx in range(0, self._n_layers):
#             if layer_idx < 1:
#                 n_features_in = n_features
#                 n_features_out = n_features + n_channels
#                 # self._layers.append(GCNBlock(n_features_in, n_features_out))
#                 self._layers.extend(self.GNNBlock(n_features, n_features_in, n_features_out))
#             else:
#                 n_features_in = n_features_out
#                 n_features_out = n_features_in + n_channels
#                 # self._layers.append(GCNBlock(n_features_in, n_features_out))
#                 self._layers.extend(self.GNNBlock(n_features_in, n_features_in, n_features_out))

#         self._stacked_layers = Sequential('x, edge_index, batch', self._layers)

#         self._linear1 = [
#             (Linear(
#                 in_features=n_features_out,
#                 out_features=n_features_out//4,
#             )),
#             'x -> x'
#         ]

#         self._linear2 = [
#             (Linear(
#                 in_features=n_features_out//4,
#                 out_features=n_classes,
#             )),
#             'x -> x'
#         ]

#         self._linear_layers = Sequential('x', [self._linear1, self._linear2])
#         # self._layers.extend([self._linear1, self._linear2])
    
#     def GNNBlock(self, n_channels, n_features_in, n_features_out):
#         _block = [
#             (BatchNorm1d(n_channels), 'x -> x'),
#             (GCNConv(in_channels=n_features_in, out_channels=n_features_out), 'x, edge_index -> x'),
#             (ReLU(inplace=True), 'x -> x'),
#         ]
#         return _block

#     def forward(self, x, edge_index, batch):
#         # for i in range(self._n_layers):
#         #     x = self._stacked_layers[i](x, edge_index)
#         # p = global_mean_pool(x, batch)
#         # z = self._linear1(p)
#         # z = self._linear2(z)
#         x = self._stacked_layers(x, edge_index, batch)
#         p = global_mean_pool(x, batch)
#         out = self._linear_layers(p)
#         # out = self._stacked_layers(x, edge_index, batch)
#         return out

# class GNNConcatSkipConnections(torch.nn.Module):
#     """
#     TODO
#     """
#     def __init__(self):
#         super(GNNConcatSkipConnections, self).__init__()

#         self._model = Sequential(
#             'x, edge_index, batch',
#             [
#                 (BatchNorm1d(7), 'x, edge_index -> x'),
#                 (GCNConv(in_channels=7, out_channels=71), 'x, edge_index -> x'),
#                 (ReLU(inplace=True), 'x, edge_index -> x'),
#                 (BatchNorm1d(71), 'x, edge_index -> x'),
#                 (GCNConv(in_channels=71, out_channels=135), 'x, edge_index -> x'),
#                 (ReLU(inplace=True), 'x, edge_index -> x'),
#                 (BatchNorm1d(135), 'x, edge_index -> x'),
#                 (GCNConv(in_channels=135, out_channels=199), 'x, edge_index -> x'),
#                 (ReLU(inplace=True), 'x, edge_index -> x'),
#                 (global_mean_pool, 'x, batch -> x'),
#                 (Linear(199, 199//4), 'x -> x'),
#                 (Linear(199//4, 2), 'x -> x'),
#             ]
#         )
    
#     def forward(self, x, edge_index, batch):
#         out = self._model(x, edge_index, batch)
#         return out