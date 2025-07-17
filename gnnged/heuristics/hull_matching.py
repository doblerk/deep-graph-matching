import sys
import json
import logging
import numpy as np
from pathlib import Path
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch_geometric.datasets import TUDataset
from scipy.spatial.distance import pdist, squareform
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


def get_convex_hull_class(dim):
    if dim == 2:
        return ConvexHull2D
    elif dim == 3:
        return ConvexHull3D
    else:
        return ConvexHullBase
    

def calc_matrix_distances(dataset, node_embeddings, output_dir, dims):
    logging.info("Starting distance computation...")
    
    for dim in dims:
        t0 = time()
        hull_class = get_convex_hull_class(dim)

        logging.info(f"Processing convex hulls for dimension: {dim} using {hull_class.__name__}")

        feature_dim = len(hull_class(node_embeddings[dim][0]).compute_all())
        convex_hull_matrix = np.zeros((len(dataset), feature_dim), dtype=np.float32)
        valid_indices = []
        
        for i, embedding in node_embeddings[dim].items():
            try:
                features = hull_class(embedding).compute_all()
                convex_hull_matrix[i] = np.array(features, dtype=np.float32)
                valid_indices.append(i)
            except ValueError as e:
                logging.warning(f"Skipping graph {i} at dim {dim}: {e}")
                continue

        normalized = normalize(convex_hull_matrix, axis=0, norm='max')
        distances = squareform(pdist(normalized, metric='euclidean'))

        duration = time() - t0
        logging.info(f"Finished computation in {duration} seconds for dimension {dim}")

        distance_file = output_dir / f'distances_{dim}d'
        index_file = output_dir / f'valid_indices_{dim}d.json'

        np.save(distance_file, distances)
        with open(index_file, 'w') as f:
            json.dump(valid_indices, f)
        
        logging.info(f"Saved distance matrix to {distance_file.resolve()}")
        logging.info(f"Saved valid indices to {index_file.resolve()}")


def main(config):
    """
    Computes all pairwise distances between convex hull features of node embeddings for graphs.
    """
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'log_hull_matching.txt'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading dataset and node embeddings...")
    dataset = TUDataset(root=config['dataset_dir'], name=config['dataset_name'])
    
    node_embeddings = load_embeddings(output_dir / 'node_embeddings.h5', len(dataset))
    dims = (2, 3)
    reduced_node_embeddings = reduce_embedding_dimensionality(node_embeddings, dims)

    calc_matrix_distances(dataset, reduced_node_embeddings, output_dir, dims)


if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)