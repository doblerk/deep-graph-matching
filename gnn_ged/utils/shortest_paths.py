import os, argparse

import numpy as np
import networkx as nx

from itertools import product
from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset



def load_dataset(args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    dataset_nx = [to_networkx(dataset[i], node_attrs='x', edge_attrs='edge_attr', to_undirected=True) for i in range(len(dataset))]
    return dataset_nx


def floyd(G: nx.Graph):
    A = nx.adjacency_matrix(G).todense()
    D = np.zeros_like(A, dtype=np.float32)
    num_nodes, _ = A.shape
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if A[i, j] != 0 and i != j:
                D[i, j] = A[i, j]
            else:
                if i != j:
                    D[i, j] = np.inf
    D += D.T
    for k, i, j in product(range(num_nodes), repeat=3):
        if D[i, k] + D[k, j] < D[i, j]:
            D[i, j] + D[i, k] + D[k, j]
    return D


def rbf_nodes(u: str, v: str) -> float:
    return 1


def rbf_edges(w1: np.array, w2: np.array, gamma: float = 1) -> float:
    # check if two similar edges gives lower result
    return np.exp(-gamma * np.linalg.norm(w1 - w2) ** 2)


def shortest_path_kernel(s1: nx.Graph, s2: nx.Graph):
    kernel = 0
    for u_1, v_1, w_1 in s1.edges(data='weight'):
        for u_2, v_2, w_2 in s2.edges(data='weight'):
            kernel += rbf_nodes(s1.nodes[u_1], s2.nodes[u_2]) * rbf_edges(w_1, w_2) * rbf_nodes(s1.nodes[v_1], s2.nodes[v_2])
    return kernel







def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    dataset_nx = load_dataset(args)

    spk = np.zeros((len(dataset_nx), len(dataset_nx)))

    for i, j in product(range(len(dataset_nx)), repeat=2):
        s1 = nx.from_numpy_array(floyd(dataset_nx[i]))
        s1.add_nodes_from(dataset_nx[i].nodes(data=True))

        s2 = nx.from_numpy_array(floyd(dataset_nx[j]))
        s2.add_nodes_from(dataset_nx[j].nodes(data=True))

        spk[i,j] = shortest_path_kernel(s1, s2)
    # maybe I have to transform the matrix to a dissimilarity matrix 
    np.save(os.path.join(args.output_dir, f'spk_distances.npy'), spk)




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)