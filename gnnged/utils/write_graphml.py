import os
import argparse
import networkx as nx
from torch_geometric.datasets import TUDataset


def main(args):

    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    for i in range(len(dataset)):
        graph = dataset[i]  # Get the i-th graph data
        
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_idx in range(graph.num_nodes):
            node_attrs = ','.join(map(str, graph.x[node_idx].tolist()))
            G.add_node(node_idx, attr=node_attrs) 
        
        # Add edges
        edge_index = graph.edge_index.t().tolist()
        G.add_edges_from(edge_index)
        
        # Export to GraphML
        graphml_file = f'graph_{i}.graphml'
        nx.write_graphml(G, os.path.join(args.output_dir, graphml_file))


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)