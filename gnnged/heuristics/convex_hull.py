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


class ConvexHullBase(ConvexHull):

    def __init__(self, points):
        super().__init__(self, points, qhull_options='QJ')
        self.points = points
        self._dimension = points.shape[1]
        self._volume = self.volume  # Area in 2D, volume in 3D
        self._centroid = None
        self._diameter = None
        self._perimeter = None
        self._pca = None
        
    def calc_centroid(self):
        if self._centroid is None:
            self._centroid = [
                np.mean(self.points[self.vertices, i])
                for i in range(self._dimension)
            ]
        return self._centroid

    def calc_hull_size(self):
        return len(self.vertices)

    def calc_perimeter(self):
        # Note: in dimension > 2, this is a looped sum of vertex distances, not the true surface perimeter
        if self._perimeter is None:
            vertices = self.vertices.tolist() + [self.vertices[0]]
            self._perimeter = sum(
                euclidean(x, y) 
                for x, y in zip(self.points[vertices], self.points[vertices[1:]])
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
    
    def calc_compactness(self):
        radius = self.calc_diameter() / 2
        vol_n_sphere = (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
        return self._volume / vol_n_sphere if vol_n_sphere > 0 else 0
    
    def _compute_pca(self):
        if self._pca is None:
            self._pca = PCA(n_components=self.points.shape[1])
            self._pca.fit(self.points[self.vertices])
    
    def calc_major_minor_axes(self):
        self._compute_pca()
        axis_lengths = np.sqrt(self._pca.explained_variance_)
        axis_directions = self._pca.components_
        return {
            'axis_lengths': axis_lengths,         # e.g., [major, ..., minor]
            'axis_directions': axis_directions    # unit vectors for each axis
        }
    
    def calc_elongation(self):
        self._compute_pca()
        axis_lengths = np.sqrt(self._pca.explained_variance_)
        return axis_lengths[0] / axis_lengths[-1] if axis_lengths[-1] > 0 else float('inf')
    
    def calc_spatial_distribution(self):
        distances = np.linalg.norm(self.points - self._centroid, axis=1)
        return np.mean(distances)

    # def calc_isoperimetric_quotient(self):
    #     perimeter = self.calc_perimeter()
    #     area = self.calc_volume()
    #     r_circle = perimeter / (2 * np.pi)
    #     area_circle = np.pi * r_circle**2
    #     return area / area_circle

    def calc_minimum_bounding_sphere(self):
        radius = self.calc_diameter() / 2
        vol_n_sphere = (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
        compactness = self.calc_volume() / vol_n_sphere if vol_n_sphere > 0 else 0
        return {
            'sphere_radius': radius,
            'sphere_volume': vol_n_sphere,
            'compactness': compactness
        }
    
    def calc_edge_statistics(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        edges = [euclidean(x, y) for x, y in zip(self.points[vertices], self.points[vertices[1:]])]
        return {
            'mean_edge_length': np.mean(edges),
            'shortest_edge': min(edges),
            'longest_edge': max(edges),
            'ratio_lengths': min(edges) / max(edges)
        }
    
    def calc_point_density(self):
        return self.calc_hull_size() / self.calc_volume()

    def calc_relative_shape(self):
        return self.calc_perimeter() / self.calc_volume()

    def calc_relative_spatial(self):
        return self.calc_diameter() / self.calc_volume()

    # def calc_all_feats(self, config):
    #     feats = []
    #     if config.get('use_centroid', True):
    #         feats.extend([*self.calc_centroid()])
    #     if config.get('use_hull_size', True):
    #         feats.append(self.calc_hull_size())
    #     if config.get('use_perimeter', True):
    #         feats.append(self.calc_perimeter())
    #     if config.get('use_area', True):
    #         feats.append(self.calc_volume())
    #     if config.get('use_diameter', True):
    #         feats.append(self.calc_diameter())
    #     if config.get('use_iq', True):
    #         feats.append(self.calc_isoperimetric_quotient())
    #     if config.get('use_mbs', True):
    #         mbs = self.calc_minimum_bounding_sphere()
    #         feats.extend([mbs['sphere_radius'], 
    #                       mbs['sphere_volume'], 
    #                       mbs['compactness']])
    #     if config.get('use_edge_stats', True):
    #         edge_stats = self.calc_edge_statistics()
    #         feats.extend([edge_stats['mean_edge_length'], 
    #                       edge_stats['shortest_edge'],
    #                       edge_stats['longest_edge'], 
    #                       edge_stats['ratio_lengths']])
    #     if config.get('use_point_density', True):
    #         feats.append(self.calc_point_density())
    #     if config.get('use_rel_shape', True):
    #         feats.append(self.calc_relative_shape())
    #     if config.get('use_rel_spatial', True):
    #         feats.append(self.calc_relative_spatial())
    #     return feats

    def compute_all(self, config=None):
        if config is None:
            config = {}

        feats = []
        feats.extend(self.calc_centroid())
        feats.append(self.calc_hull_size())
        feats.append(self.calc_perimeter())
        feats.append(self.calc_volume())
        feats.append(self.calc_diameter())
        feats.append(self.calc_compactness())

        mbs = self.calc_minimum_bounding_sphere()
        feats.extend([mbs['sphere_radius'], mbs['sphere_volume'], mbs['compactness']])

        edge_stats = self.calc_edge_statistics()
        feats.extend(edge_stats.values())

        feats.append(self.calc_point_density())
        feats.append(self.calc_relative_shape())
        feats.append(self.calc_relative_spatial())

        try:
            feats.append(self.calc_elongation())
            if self._dimension in [2, 3]:
                feats.append(self.calc_eccentricity())
        except:
            feats.append(None)

        return feats


class ConvexHull2D(ConvexHullBase):
    
    def __init__(self, points):
        super().__init__(points, qhull_options='QJ')
        self._area = self._volume
    
    def calc_circularity(self):
        perimeter = self.calc_perimeter()
        return (4 * np.pi * self._area) / (perimeter ** 2) if perimeter > 0 else 0
    
    def calc_eccentricity(self):
        axes = self.calc_major_minor_axes()['axis_lengths']
        a, b = axes[0], axes[1]
        return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0
    
    def compute_all(self, config=None):
        feats = super().compute_all(config)
        feats.append(self.calc_circularity())
        feats.append(self.calc_eccentricity())
        return feats

class ConvexHull3D(ConvexHullBase):

    def __init__(self, points):
        super().__init__(points, qhull_options='QJ')
        self._volume = self._volume
    
    def calc_sphericity(self):
        surface_area = self.area
        return (np.pi ** (1 / 3)) * ((6 * self._volume) ** (2 / 3)) / surface_area if surface_area > 0 else 0
    
    def calc_eccentricity(self):
        axes = self.calc_major_minor_axes()['axis_lengths']
        a, b = axes[0], axes[1]
        return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0
    
    def compute_all(self, config=None):
        feats = super().compute_all(config)
        feats.append(self.calc_sphericity())
        feats.append(self.calc_eccentricity())
        return feats


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