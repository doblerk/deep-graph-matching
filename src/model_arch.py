import os
import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool


class GCN(torch.nn.Module):
    '''
    A class defining the Graph Convolutional Network

    Args:
        n_features: number of features per node
        n_channels: number of channels
        n_classes: number of classes
    '''
    def __init__(self, n_features, n_channels, n_classes):
        super(GCN, self).__init__()
        
        torch.manual_seed(4038)
        
        self._conv1 = GCNConv(
            in_channels=n_features,
            out_channels=n_channels,
        )

        self._conv2 = GCNConv(
            in_channels=n_channels,
            out_channels=n_channels,
        )

        self._conv3 = GCNConv(
            in_channels=n_channels,
            out_channels=n_channels,
        )

        self._conv4 = GCNConv(
            in_channels=n_channels,
            out_channels=n_channels,
        )

        self._linear1 = Linear(
            in_features=n_channels,
            out_features=n_classes,
        )
    
    def forward(self, x, edge_index, batch):
        h1 = self._conv1(x, edge_index)
        h1 = F.relu(h1)
        h2 = self._conv2(h1, edge_index)
        h2 = F.relu(h2)
        h3 = self._conv3(h2, edge_index)
        h3 = F.relu(h3)
        h4 = self._conv4(h3, edge_index)
        h4 = F.relu(h4)
        
        p = global_add_pool(h4, batch)
        
        z = F.dropout(p, p=0.2, training=self.training)
        z = self._linear1(z)
        
        return h1, h2, h3, h4, z





class GINLayer(torch.nn.Module):
    '''
    A class defining the Graph Isomorphism layer

    Args:
        input_dim: number of features per node
        hidden_dim: number of channels
    '''
    def __init__(self, input_dim, hidden_dim) -> None:
        super(GINLayer, self).__init__()
        
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
    '''
    A class defining the Graph Isomorphism Network

    Args:
        input_dim: number of features per node
        hidden_dim: number of channels
        n_classes: number of classes
    '''
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers):
        super(GINModel, self).__init__()

        torch.manual_seed(4030)

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
    '''
    A class defining the Graph Convolutional Network using concatenated skip connections

    Args:
        n_features: number of features per node
        n_channels: number of channels
    '''
    def __init__(self, n_features, n_channels):
        super(GCNBlock, self).__init__()
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
    '''
    A class defining the stack of multiple GCN blocks

    Args:
        n_features: number of features per node
        n_channels: number of channels
        n_classes: number of classes
        n_layers: number of layers
    '''
    def __init__(self, n_features, n_channels, n_classes, n_layers):
        super(GCNBlockStack, self).__init__()
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
    '''
    TODO
    '''
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
