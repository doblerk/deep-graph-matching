import sys
import json
import h5py
import logging
import numpy as np
from pathlib import Path
from time import time
from scipy.special import gamma
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch_geometric.datasets import TUDataset

from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist, squareform

from gnnged.heuristics.data_utils import load_embeddings, reduce_embedding_dimensionality
from gnnged.heuristics.hull_utils import ConvexHullBase, ConvexHull2D, ConvexHull3D


# def calc_matrix_distances(args):

#     dataset = dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

#     embeddings = load_embeddings(args, len(dataset))

#     min_graph = get_min_graph(dataset)

#     dims = list(range(2, min_graph)) # min_graph - 1 but range() stops at end - 1

#     reduced_embeddings = reduce_embedding_dimensionality(embeddings, dims)

#     """
#         -> [Cx, Cy, ..., Rsh, Rsp]
#     """

#     feats = ['use_centroid', 'use_hull_size', 'use_perimeter', 'use_area',
#              'use_diameter', 'use_iq', 'use_mbs', 'use_edge_stats',
#              'use_point_density', 'use_rel_shape', 'use_rel_spatial']
    
#     all_feats_true = {feat: True for feat in feats}
    
#     leave_one_out_feats = [
#         {feat: (feat != leave_out) for feat in feats}
#         for leave_out in feats
#     ]
    
#     ablation_configs = [all_feats_true] + leave_one_out_feats

#     for dim in dims:

#         distance_matrices = []

#         with open(os.path.join(args.output_dir, 'computation_time_convhull.txt'), 'a') as file:
#             file.write(f'Computation times for dimension {dim}:\n')

#         for config_idx, config in enumerate(ablation_configs):

#             convex_hulls = dict.fromkeys(range(len(embeddings)), list())
            
#             t0 = time()
            
#             for graph_idx, _ in convex_hulls.items():
#                 convex_hulls[graph_idx] = ConvexHullChild(reduced_embeddings[dim][graph_idx]).calc_all_feats(config)
            
#             convex_hulls_stacked = np.vstack(list(convex_hulls.values()))
#             convex_hulls_stacked_normalized = normalize(convex_hulls_stacked, axis=0, norm='max')
#             matrix_distances = squareform(pdist(convex_hulls_stacked_normalized, metric='euclidean'))

#             t1 = time()
#             computation_time = t1 - t0

#             with open(os.path.join(args.output_dir, 'computation_time_convhull.txt'), 'a') as file:
#                 file.write(f'   Computation times for config {config_idx:2}: {computation_time}\n')

#             distance_matrices.append(matrix_distances)        
        
#         with h5py.File(os.path.join(args.output_dir, f'distances_dim_{dim}.h5'), 'w') as f:
#             for config, matrix in enumerate(distance_matrices, start=0):
#                 f.create_dataset(f'config_{config}', data=matrix)

def calc_matrix_distances(dataset, node_embeddings, output_dir, dims):
    logging.info("Starting distance computation...")
    
    for dim in dims:

        convex_hulls = dict.fromkeys(range(len(dataset)), list())

        t0 = time()

        if dim == 2:
            for graph_idx, _ in convex_hulls.items():
                convex_hulls[graph_idx] = ConvexHull2D(node_embeddings[dim][graph_idx]).compute_all(config)
        elif dim == 3:
            for graph_idx, _ in convex_hulls.items():
                convex_hulls[graph_idx] = ConvexHull3D(node_embeddings[dim][graph_idx]).compute_all(config)
        else:
            for graph_idx, _ in convex_hulls.items():
                convex_hulls[graph_idx] = ConvexHullBase(node_embeddings[dim][graph_idx]).compute_all(config)
        
        convex_hulls_stacked = np.vstack(list(convex_hulls.values()))
        convex_hulls_stacked_normalized = normalize(convex_hulls_stacked, axis=0, norm='max')
        matrix_distances = squareform(pdist(convex_hulls_stacked_normalized, metric='euclidean'))

        duration = time() - t0
        logging.info(f"Finished computation in {duration} seconds for dimension {dim}")

        distance_file = output_dir / f'distances_{dim}d.py'
        np.save(distance_file, matrix_distances)
        logging.info(f"Saved distance matrix to {distance_file.resolve()}")


def main(config):
    """
    Computes all pairwise distance between each pair of graph node embeddings to yield a dissimilarity matrix.

    Args:
        config: commandline arguments (path to dataset directory, dataset name, path to node embeddings, path to output directory)
    """
    # Setup logging
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'log_hull_matching.txt'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    dataset = TUDataset(root=config['dataset_dir'], name=config['dataset_name'])

    dims = (2, 3)
    
    node_embeddings = load_embeddings(output_dir / 'node_embeddings.h5', len(dataset))
    reduced_node_embeddings = reduce_embedding_dimensionality(node_embeddings, dims)

    calc_matrix_distances(dataset, reduced_node_embeddings, output_dir, dims)


if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)