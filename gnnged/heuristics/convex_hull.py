import os
import h5py
import argparse

import numpy as np

from time import time
from scipy.special import gamma
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch_geometric.datasets import TUDataset

from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist, squareform


class ConvexHullChild(ConvexHull):

    def __init__(self, points):
        ConvexHull.__init__(self, points, qhull_options='QJ')
        self._centroid = None
        self._perimeter = None
        self._volume = self.volume
        self._diameter = None

    def calc_centroid(self):
        if self._centroid is None:
            self._centroid = [
                np.mean(self.points[self.vertices, i])
                for i in range(self.points.shape[1])
            ]
        return self._centroid

    def calc_hull_size(self):
        return len(self.vertices)

    def calc_perimeter(self):
        if self._perimeter is None:
            vertices = self.vertices.tolist() + [self.vertices[0]]
            self._perimeter = sum(
                euclidean(x, y) for x, y in zip(self.points[vertices], self.points[vertices[1:]])
            )
        return self._perimeter

    def calc_volume(self):
        return self._volume
    
    def calc_diameter(self):
        if self._diameter is None:
            self._diameter = np.max(
                pdist(self.points[self.vertices], metric='euclidean')
            )
        return self._diameter

    def calc_isoperimetric_quotient(self):
        perimeter = self.calc_perimeter()
        area = self.calc_volume()
        r_circle = perimeter / (2 * np.pi)
        area_circle = np.pi * r_circle**2
        return area / area_circle

    def calc_minimum_bounding_sphere(self):
        dimension = self.points.shape[1]
        radius = self.calc_diameter() / 2
        n_sphere_volume = (np.pi ** (dimension / 2) * radius ** dimension) / gamma((dimension / 2) + 1)
        compactness = self.calc_volume() / n_sphere_volume if n_sphere_volume > 0 else 0
        return {
            'sphere_radius': radius,
            'sphere_volume': n_sphere_volume,
            'compactness': compactness
        }
    
    def calc_edge_statistics(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        edge_lengths = [euclidean(x, y) for x, y in zip(self.points[vertices], self.points[vertices[1:]])]
        return {
            'mean_edge_length': np.mean(edge_lengths),
            'shortest_edge': min(edge_lengths),
            'longest_edge': max(edge_lengths),
            'ratio_lengths': min(edge_lengths) / max(edge_lengths)
        }
    
    def calc_point_density(self):
        return self.calc_hull_size() / self.calc_volume()

    def calc_relative_shape(self):
        return self.calc_perimeter() / self.calc_volume()

    def calc_relative_spatial(self):
        return self.calc_diameter() / self.calc_volume()

    def calc_all_feats(self, config):
        feats = []
        if config.get('use_centroid', True):
            feats.extend([*self.calc_centroid()])
        if config.get('use_hull_size', True):
            feats.append(self.calc_hull_size())
        if config.get('use_perimeter', True):
            feats.append(self.calc_perimeter())
        if config.get('use_area', True):
            feats.append(self.calc_volume())
        if config.get('use_diameter', True):
            feats.append(self.calc_diameter())
        if config.get('use_iq', True):
            feats.append(self.calc_isoperimetric_quotient())
        if config.get('use_mbs', True):
            mbs = self.calc_minimum_bounding_sphere()
            feats.extend([mbs['sphere_radius'], 
                          mbs['sphere_volume'], 
                          mbs['compactness']])
        if config.get('use_edge_stats', True):
            edge_stats = self.calc_edge_statistics()
            feats.extend([edge_stats['mean_edge_length'], 
                          edge_stats['shortest_edge'],
                          edge_stats['longest_edge'], 
                          edge_stats['ratio_lengths']])
        if config.get('use_point_density', True):
            feats.append(self.calc_point_density())
        if config.get('use_rel_shape', True):
            feats.append(self.calc_relative_shape())
        if config.get('use_rel_spatial', True):
            feats.append(self.calc_relative_spatial())
        return feats


def load_embeddings(args, dataset_size):
    """Loads the train and test embeddings"""
    with h5py.File(args.node_embeddings, 'r') as f:
        node_embeddings = {i:np.array(f[f'embedding_{i}']) for i in range(dataset_size)}
    return node_embeddings


def reduce_embedding_dimensionality(embeddings, dims):
    """Reduces the dimensionality of each node embedding"""
    return {
        dim: {i: PCA(n_components=dim).fit_transform(embedding) for i, embedding in embeddings.items()}
        for dim in dims
    }


def get_min_graph(dataset):
    """Returns the size of the smaller graph"""
    return min(data.x.shape[0] for data in dataset)


def calc_matrix_distances(args):

    dataset = dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    embeddings = load_embeddings(args, len(dataset))

    min_graph = get_min_graph(dataset)

    dims = list(range(2, min_graph)) # min_graph - 1 but range() stops at end - 1

    reduced_embeddings = reduce_embedding_dimensionality(embeddings, dims)

    """
        -> [Cx, Cy, ..., Rsh, Rsp]
    """

    feats = ['use_centroid', 'use_hull_size', 'use_perimeter', 'use_area',
             'use_diameter', 'use_iq', 'use_mbs', 'use_edge_stats',
             'use_point_density', 'use_rel_shape', 'use_rel_spatial']
    
    all_feats_true = {feat: True for feat in feats}
    
    leave_one_out_feats = [
        {feat: (feat != leave_out) for feat in feats}
        for leave_out in feats
    ]
    
    ablation_configs = [all_feats_true] + leave_one_out_feats

    for dim in dims:

        distance_matrices = []

        with open(os.path.join(args.output_dir, 'computation_time_convhull.txt'), 'a') as file:
            file.write(f'Computation times for dimension {dim}:\n')

        for config_idx, config in enumerate(ablation_configs):

            convex_hulls = dict.fromkeys(range(len(embeddings)), list())
            
            t0 = time()
            
            for graph_idx, _ in convex_hulls.items():
                convex_hulls[graph_idx] = ConvexHullChild(reduced_embeddings[dim][graph_idx]).calc_all_feats(config)
            
            convex_hulls_stacked = np.vstack(list(convex_hulls.values()))
            convex_hulls_stacked_normalized = normalize(convex_hulls_stacked, axis=0, norm='max')
            matrix_distances = squareform(pdist(convex_hulls_stacked_normalized, metric='euclidean'))

            t1 = time()
            computation_time = t1 - t0

            with open(os.path.join(args.output_dir, 'computation_time_convhull.txt'), 'a') as file:
                file.write(f'   Computation times for config {config_idx:2}: {computation_time}\n')

            distance_matrices.append(matrix_distances)        
        
        with h5py.File(os.path.join(args.output_dir, f'distances_dim_{dim}.h5'), 'w') as f:
            for config, matrix in enumerate(distance_matrices, start=0):
                f.create_dataset(f'config_{config}', data=matrix)
        

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--node_embeddings', type=str, help='Path to node embeddings file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    """
    Computes all pairwise distance between each pair of graph node embeddings to yield a dissimilarity matrix.

    Args:
        args: commandline arguments (path to dataset directory, dataset name, path to node embeddings, path to output directory)
    """
    calc_matrix_distances(args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)