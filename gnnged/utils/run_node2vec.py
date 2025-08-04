import h5py
import argparse
import numpy as np
from time import time
from pathlib import Path
from node2vec import Node2Vec
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import Constant
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform


def run_node2vec(dataset):
    node_embeddings = {}
    
    p, q = 1.0, 1.0
    for i in range(len(dataset)):
        G = to_networkx(dataset[i], node_attrs='x', to_undirected=True)
        avg_deg = compute_avg_degree(G)
        wl = max(10, int(avg_deg / 2))
        node2vec = Node2Vec(
            G, 
            dimensions=64, 
            walk_length=wl, 
            num_walks=10, 
            p=p, 
            q=q,
            seed=42, 
            workers=4,
            quiet=True
        )

        model = node2vec.fit(window=10, min_count=1)
        
        nodes = list(G.nodes())
        embeddings = np.array([model.wv[str(node)] for node in nodes])

        node_embeddings[i] = embeddings
    
    return node_embeddings


def compute_avg_degree(graph):
    degrees = [deg for _, deg in graph.degree()]
    return np.mean(degrees)


def run_pairwise_distances(node_embeddings):
    normalized = normalize(node_embeddings, axis=0, norm='l2')
    distances = squareform(pdist(normalized, metric='euclidean'))
    return distances


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    if not hasattr(dataset[0], 'x') or dataset[0].x is None:
        dataset.transform = Constant(value=1.0)

    t0 = time()
    node_embeddings = run_node2vec(dataset)
    duration = time() - t0
    print(f'Duration time for dataset {args.dataset_name}: {duration}')

    node_embeddings_file = Path(args.output_dir) / f'node2vec_embeddings.h5'
    with h5py.File(node_embeddings_file, 'w') as f:
        for i, mbddg in node_embeddings.items():
            f.create_dataset(f'embedding_{i}', data=mbddg)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)