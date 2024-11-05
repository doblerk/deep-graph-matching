import os
import h5py
import argparse
import numpy as np

from time import time
from itertools import combinations

from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset

from gnn_ged.assignment.calc_assignment import NodeAssignment
from gnn_ged.edit_cost.calc_edit_cost import EditCost


def load_dataset(args):
    """Loads the dataset from TUDataset and converts it into NetworkX"""
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    dataset_nx = [to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))]
    return dataset_nx


def load_embeddings(args, dataset_size):
    """Loads the train and test embeddings"""
    with h5py.File(args.node_embeddings, 'r') as f:
        node_embeddings = [np.array(f[f'embedding_{i}']) for i in range(dataset_size)]
    return node_embeddings


def calc_matrix_distances(args):
    """Calculates the matrix of distances"""
    dataset_nx = load_dataset(args)
    
    node_embeddings = load_embeddings(args, len(dataset_nx))

    matrix_distances = np.zeros(shape=(len(node_embeddings), len(node_embeddings)), dtype=np.int32)
    
    t0 = time()

    for i, j in combinations(list(range(len(dataset_nx))), r=2):

        g1_nx = dataset_nx[i]
        g2_nx = dataset_nx[j]

        if g1_nx.number_of_nodes() <= g2_nx.number_of_nodes():
            # heuristic -> the smaller graph is always the source graph
            source_embedding = node_embeddings[i]
            target_embedding = node_embeddings[j]
            source_graph = g1_nx
            target_graph = g2_nx
        else:
            source_embedding = node_embeddings[j]
            target_embedding = node_embeddings[i]
            source_graph = g2_nx
            target_graph = g1_nx

        node_assignment = NodeAssignment(source_embedding, target_embedding)

        embedding_distances = node_assignment.compute_embedding_distances()

        assignment = node_assignment.compute_node_assignment(embedding_distances)

        edit_cost = EditCost(assignment, source_graph, target_graph)

        node_cost = edit_cost.compute_cost_node_edit()
        edge_cost = edit_cost.compute_cost_edge_edit()
        
        matrix_distances[i,j] = node_cost + edge_cost

    matrix_distances += matrix_distances.T

    t1 = time()
    computation_time = t1 - t0
    
    with open(os.path.join(args.output_dir, 'computation_time.txt'), 'a') as file:
        file.write(str(computation_time) + '\n')
    
    np.save(os.path.join(args.output_dir, f'distances.npy'), matrix_distances)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--node_embeddings', type=str, help='Path to node embeddings file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    """
    Computes all pairwise distances between every pair of graphs to yield a dissimilarity matrix.

    Args:
        args: command-line arguments (path to dataset directory, dataset name, path to node embeddings, and path to output directory).
    """
    calc_matrix_distances(args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)