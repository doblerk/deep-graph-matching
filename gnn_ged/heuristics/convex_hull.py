import os
import h5py
import argparse

import numpy as np

from time import time
from itertools import product
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize 
from torch_geometric.datasets import TUDataset

from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist, squareform


# https://stackoverflow.com/questions/26408110/why-does-qhull-error-when-computing-convex-hull-of-a-few-points
# https://gist.github.com/Vini2/2d35132f70ee18298fdea142b5530a52
# https://medium.com/@errazkim/computing-the-convex-hull-in-python-60a6087e0faa
# The expected number of vertices in the convex hull is O(n^(m−1/m+1)) for n→∞. It was proved in 1970 by H. Raynaud in "Sur l’enveloppe convex des nuages de points aleatoires dans Rn".
# https://locationtech.github.io/jts/javadoc/org/locationtech/jts/algorithm/MinimumBoundingCircle.html


class ConvexHullChild(ConvexHull):

    def __init__(self, points):
        ConvexHull.__init__(self, points, qhull_options='QJ')

    def calc_centroid(self):
        centroid = [np.mean(self.points[self.vertices, i]) for i in range(self.points.shape[1])]
        return centroid

    def calc_perimeter(self):
        vertices = self.vertices.tolist() + [self.vertices[0]] # close the loop by adding the first vertex
        perimeter = sum(euclidean(x, y) for x, y in zip(self.points[vertices], self.points[vertices[1:]]))
        return perimeter
    
    def calc_volume(self):
        return self.volume
    
    def calc_relative_shape(self):
        return self.calc_perimeter() / self.calc_volume()
    
    def calc_hull_size(self):
        return len(self.vertices)

    def calc_diameter(self):
        diameter = np.max(pdist(self.points[self.vertices], metric='euclidean'))
        return diameter

    def calc_relative_spatial(self):
        return self.calc_diameter() / self.calc_volume()
    
    def calc_edge_statistics(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        edge_lengths = [euclidean(x, y) for x, y in zip(self.points[vertices], self.points[vertices[1:]])]
        return {
            'mean_edge_length': np.mean(edge_lengths),
            'variance_edge_length': np.var(edge_lengths),
            'shortest_edge': min(edge_lengths),
            'longest_edge': max(edge_lengths),
            'ratio_lengths': min(edge_lengths) / max(edge_lengths)
        }

    def calc_isoperimetric_quotient(self):
        area = self.calc_volume()
        perim = self.calc_perimeter()
        r_circle = perim / 2 * np.pi
        area_circle = np.pi * r_circle**2
        return area / area_circle
    
    def calc_minimum_bounding_circle(self):
        diameter = self.calc_diameter()
        radius = diameter / 2
        area_circle = np.pi * radius**2
        return {
            'circle_radius': radius,
            'circle_area': area_circle,
            'compactness': self.calc_volume / area_circle # Reock Compactness
        }
    
    def calc_turning_functions(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        turning_angles = [
            np.arctan2(
                self.points[vertices[i + 1]][1] - self.points[vertices[i]][1],
                self.points[vertices[i + 1]][0] - self.points[vertices[i]][0]
            )
            for i in range(len(self.vertices))
        ]
        return turning_angles

    def calc_aspect_ratio(self):
        pca = np.linalg.svd(self.points[self.vertices] - np.mean(self.points[self.vertices], axis=0))
        return pca[1][0] / pca[1][-1]  # Ratio of largest to smallest singular value

    def calc_all_feats(self, 
                       use_centroid=True,
                       use_area=True,
                       use_perimeter=True,
                       use_hull_size=True,
                       use_diameter=True,
                       use_iq=True,
                       use_rel_shape=True,
                       use_rel_spatial=True):
        feats = []
        if use_centroid:
            feats.append(*self.calc_centroid())
        if use_area:
            feats.append(self.calc_volume())
        if use_perimeter:
            feats.append(self.calc_perimeter())
        if use_hull_size:
            feats.append(self.calc_hull_size())
        if use_diameter:
            feats.append(self.calc_diameter())
        if use_iq:
            feats.append(self.calc_isoperimetric_quotient())
        if use_rel_shape and use_area and use_perimeter:
            feats.append(self.calc_perimeter() / self.calc_volume())
        if use_rel_spatial and use_area and use_diameter:
            feats.append(self.calc_diameter() / self.calc_volume())
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
        -> [Cx, Cy, ..., P, V, S, D, Rsh, Rsp]
    """

    feats = ['use_centroid', 'use_area', 'use_perimeter', 'use_hull_size',
             'use_diameter', 'use_rel_shape', 'use_rel_spatial']
    
    ablation_configs = [dict(zip(feats, combo)) for combo in product([True, False], repeat=len(feats))]

    for dim in dims:

        convex_hulls = dict.fromkeys(range(len(embeddings)), list())

        t0 = time()

        for k, _ in convex_hulls.items():
            convex_hulls[k] = ConvexHullChild(reduced_embeddings[dim][k]).calc_all_feats()

        convex_hulls_stacked = np.vstack(list(convex_hulls.values()))
        convex_hulls_stacked_normalized = normalize(convex_hulls_stacked, axis=0, norm='max')
        matrix_distances = squareform(pdist(convex_hulls_stacked_normalized, metric='euclidean'))

        t1 = time()
        computation_time = t1 - t0

        with open(os.path.join(args.output_dir, 'computation_time_convhull.txt'), 'a') as file:
            file.write(f'Computation time for {dim}: {computation_time}\n')
        
        np.save(os.path.join(args.output_dir, f'distances_convhull_{dim}.npy'), matrix_distances)


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