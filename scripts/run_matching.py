import os
import json
import h5py
import logging
import numpy as np

from time import time
from itertools import combinations

from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, Constant

from gnnged.assignment.calc_assignment import NodeAssignment
from gnnged.edit_cost.calc_edit_cost import EditCost


logging.basicConfig(
    level=logging.INFO,  # Or DEBUG, WARNING, ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("run_matching.log"),
        logging.StreamHandler()
    ]
)


def load_dataset(config):
    """Loads the dataset from TUDataset and converts it into NetworkX"""
    transform = NormalizeFeatures() if config['use_attrs'] else None
    dataset = TUDataset(root=config['dataset_dir'],
                        name=config['dataset_name'],
                        use_node_attr=config['use_attrs'],
                        transform=transform)
    if 'x' not in dataset[0]:
        dataset.transform = Constant(value=1.0)
    dataset_nx = {i:to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))}
    return dataset_nx


def load_embeddings(config, dataset_size):
    """Loads the train and test embeddings"""
    with h5py.File(os.path.join(config['output_dir'], 'node_embeddings.h5'), 'r') as f:
        node_embeddings = {i:np.array(f[f'embedding_{i}']) for i in range(dataset_size)}
    return node_embeddings


def calc_matrix_distances(config):
    """Calculates the matrix of distances"""
    dataset_nx = load_dataset(config)
    
    node_embeddings = load_embeddings(config, len(dataset_nx))

    matrix_distances = np.zeros(shape=(len(node_embeddings), len(node_embeddings)), dtype=np.int32)
    
    logging.info("Starting distance computation...")
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

        cost_matrix = node_assignment.calc_cost_matrix()

        match config['method']:

            case 'lsap':
                assignment = node_assignment.calc_linear_sum_assignment(cost_matrix)
            
            case 'greedy':
                assignment = node_assignment.calc_greedy_assignment(cost_matrix)
            
            case 'flow':
                assignment = node_assignment.calc_min_cost_flow(cost_matrix)
        
        edit_cost = EditCost(assignment, source_graph, target_graph)

        node_cost = edit_cost.compute_cost_node_edit(use_attrs=config['use_attrs'])
        edge_cost = edit_cost.compute_cost_edge_edit()
        
        matrix_distances[i,j] = node_cost + edge_cost

    matrix_distances += matrix_distances.T

    t1 = time()
    computation_time = t1 - t0
    
    # with open(os.path.join(config['output_dir'], 'computation_time.txt'), 'a') as file:
    #     file.write(f"Computation time for {config['method']} : {computation_time}\n")
    logging.info(f"Computation time for {config['method']}: {computation_time:.2f} seconds")
    
    # np.save(os.path.join(config['output_dir'], f"distances_{config['method']}.npy"), matrix_distances)


def main(config):
    """
    Computes all pairwise distances between every pair of graphs to yield a dissimilarity matrix.

    Config:
        config: command-line arguments.
    """
    calc_matrix_distances(config)


if __name__ == '__main__':
    with open('params.json', 'r') as f:
        config = json.load(f)
    main(config)