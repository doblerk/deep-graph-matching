import sys
import json
import h5py
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from time import time
from itertools import combinations

from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, Constant

from gnnged.assignment.calc_assignment import NodeAssignment
from gnnged.edit_cost.calc_edit_cost import EditCost


def calc_matrix_distances(dataset_nx: dict[int, nx.Graph], node_embeddings: dict[int, np.ndarray], output_dir: Path):
    """Calculates the matrix of distances"""
    n = len(dataset_nx)
    matrix_distances = np.zeros(shape=(n, n), dtype=np.int32)
    
    logging.info("Starting distance computation...")
    t0 = time()

    for i, j in combinations(list(range(n)), r=2):
        g1_nx = dataset_nx[i]
        g2_nx = dataset_nx[j]

        if g1_nx.number_of_nodes() <= g2_nx.number_of_nodes():
            # heuristic -> the smaller graph is always the source graph
            source_embedding, target_embedding = node_embeddings[i], node_embeddings[j]
            source_graph, target_graph = g1_nx, g2_nx
        else:
            source_embedding, target_embedding = node_embeddings[j], node_embeddings[i]
            source_graph, target_graph = g2_nx, g1_nx

        node_assignment = NodeAssignment(source_embedding, target_embedding)
        cost_matrix = node_assignment.calc_cost_matrix()

        method = config['method']
        match method:
            case 'lsap':
                assignment = node_assignment.calc_linear_sum_assignment(cost_matrix)
            case 'greedy':
                assignment = node_assignment.calc_greedy_assignment(cost_matrix)
            case 'flow':
                assignment = node_assignment.calc_min_cost_flow(cost_matrix)
        
        edit_cost = EditCost(assignment, source_graph, target_graph)
        node_cost = edit_cost.compute_cost_node_edit(use_attrs=config['use_attrs'])
        edge_cost = edit_cost.compute_cost_edge_edit()

        total_cost = node_cost + edge_cost
        matrix_distances[i,j] = total_cost

    matrix_distances += matrix_distances.T

    duration = time() - t0
    logging.info(f"Finished computation in {duration} seconds for method {method}")
    
    distance_file = output_dir / f'distances_{method}.npy'
    np.save(distance_file, matrix_distances)
    logging.info(f"Saved distance matrix to {distance_file.resolve()}")


def main(config):
    """
    Computes all pairwise distances between every pair of graphs to yield a dissimilarity matrix.

    Args:
        config: command-line arguments.
    """
    # Setup logging
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'log_matching.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Load the dataset from TUDataset
    transform = NormalizeFeatures() if config['use_attrs'] else None
    dataset = TUDataset(root=config['dataset_dir'],
                        name=config['dataset_name'],
                        use_node_attr=config.get('use_attrs', False),
                        transform=transform)
    
    if not hasattr(dataset[0], 'x') or dataset[0].x is None:
        dataset.transform = Constant(value=1.0)
        logging.info("Dataset missing node features 'x', applied Constant transform.")
    
    dataset_nx = {i:to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))}

    # Load the node embeddings
    with h5py.File(output_dir / 'node_embeddings.h5', 'r') as f:
        node_embeddings = {i:np.array(f[f'embedding_{i}']) for i in range(len(dataset_nx))}
    
    # Check for unlabelled nodes
    if not hasattr(dataset[0], 'x') or dataset[0].x is None:
        dataset.transform = Constant(value=1.0)
    
    # Compute the distances
    calc_matrix_distances(dataset_nx, node_embeddings, output_dir)


if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)