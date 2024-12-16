import os
import argparse
import numpy as np

from time import time

from itertools import combinations

from grakel import GraphKernel, Graph

from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset


def load_dataset(args):
    """Loads the dataset from TUDataset and converts it into NetworkX"""
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    dataset_nx = [to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))]
    return dataset_nx


def get_adj(g):
    g_adj = []
    for i in range(len(g)):
      tmp = [0] * len(g)
      idx = list(g.adj[i].keys())
      for j in idx:
        tmp[j] = 1
      g_adj.append(tmp)
    return g_adj


def get_node_labels(g):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in dict(g.nodes(data='x')).items()}


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--kernel', type=str, choices=['sp', 'rw', 'wl'], help='Graph kernel')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):

    dataset_nx = load_dataset(args)

    match args.kernel:

        case 'sp':
            kernel = GraphKernel(kernel={'name':'shortest_path', 'algorithm_type':'floyd_warshall'}, normalize=True)
        
        case 'rw':
            kernel = GraphKernel(kernel={'name':'random_walk'}, normalize=True)
        
        case 'wl':
            kernel = GraphKernel(kernel={'name':'weisfeiler_lehman', 'n_iter':5}, normalize=True)

    similarity_matrix = np.zeros(shape=(len(dataset_nx), len(dataset_nx)), dtype=np.float32)

    adj = {}
    node_labels = {}
    for i in range(len(dataset_nx)):
      adj[i] = get_adj(dataset_nx[i])
      node_labels[i] = get_node_labels(dataset_nx[i])

    t0 = time()
    for i, j in combinations(list(range(len(dataset_nx))), r=2):

        g1_adj = adj[i]
        g1_node_labels = node_labels[i]

        g2_adj = adj[j]
        g2_node_labels = node_labels[j]

        G1 = Graph(g1_adj, g1_node_labels)
        G2 = Graph(g2_adj, g2_node_labels)

        kernel.fit_transform([G1])

        similarity_matrix[i][j] = kernel.transform([G2]).item()
    
    similarity_matrix += similarity_matrix.T

    dissimilarity_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(dissimilarity_matrix, 0.0)

    t1 = time()
    computation_time = t1 - t0

    with open(os.path.join(args.output_dir, f'computation_time_kernels.txt'), 'a') as file:
        file.write(f'Computation time for {args.kernel} kernel: {computation_time}\n')

    np.save(os.path.join(args.output_dir, f'{args.kernel}_distances.npy'), dissimilarity_matrix)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)