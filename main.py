import os
import pickle
import argparse
import numpy as np

from time import time

from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset

from gnn_ged.assignment.calc_assignment import NodeAssignment
from gnn_ged.edit_cost.calc_edit_cost import EditCost


def load_dataset(args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    dataset_nx = [to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))]
    return dataset_nx


def load_embeddings(args):
    with open(os.path.join(args.output_dir, 'train_embeddings.pkl'), 'rb') as fp:
        train_embeddings = pickle.load(fp)
    
    with open(os.path.join(args.output_dir, 'test_embeddings.pkl'), 'rb') as fp:
        test_embeddings = pickle.load(fp)
    
    return train_embeddings, test_embeddings


def calc_matrix_distances(matrix_distances,
                          dataset_nx,
                          train_idx,
                          test_idx,
                          train_embeddings,
                          test_embeddings,
                          args):
    t0 = time()
    
    for x_test in range(0, matrix_distances.shape[0]):
        
        g1_nx = dataset_nx[test_idx[x_test]]

        for y_train in range(0, matrix_distances.shape[1]):

            g2_nx = dataset_nx[train_idx[y_train]]

            if g1_nx.number_of_nodes() <= g2_nx.number_of_nodes():
                # heuristic -> the smaller graph is always the source graph
                source_embedding = test_embeddings[test_idx[x_test]]
                target_embedding = train_embeddings[train_idx[y_train]]
                source_graph = g1_nx
                target_graph = g2_nx
            else:
                source_embedding = train_embeddings[train_idx[y_train]]
                target_embedding = test_embeddings[test_idx[x_test]]
                source_graph = g2_nx
                target_graph = g1_nx
            
            node_assignment = NodeAssignment(source_embedding, target_embedding)

            embedding_distances = node_assignment.compute_embedding_distances()

            assignment = node_assignment.compute_node_assignment(embedding_distances)

            edit_cost = EditCost(assignment, source_graph, target_graph)

            node_cost = edit_cost.compute_cost_node_edit()
            edge_cost = edit_cost.compute_cost_edge_edit()
            
            matrix_distances[x_test, y_train] = node_cost + edge_cost
       
    t1 = time()
    computation_time = t1 - t0
    print('Computation time: ', computation_time)
    
    np.save(os.path.join(args.output_dir, f'distances.npy'), matrix_distances)


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    """
    Computes all pairwise distances between every pair of graphs to yield a dissimalirty matrix.

    Args:
        args: command-line arguments (path to dataset directory, dataset name, and path to output directory).
    """

    dataset_nx = load_dataset(args)
    
    train_embeddings, test_embeddings = load_embeddings(args)

    train_idx, test_idx = list(train_embeddings.keys()), list(test_embeddings.keys())

    n_train_graphs, n_test_graphs = len(train_embeddings), len(test_embeddings)

    matrix_distances = np.zeros(shape=(n_test_graphs, n_train_graphs), dtype=np.int32)

    calc_matrix_distances(matrix_distances,
                          dataset_nx,
                          train_idx,
                          test_idx,
                          train_embeddings,
                          test_embeddings,
                          args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)