import os
import h5py
import argparse

import numpy as np

from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset

from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, cdist, pdist, squareform


# https://stackoverflow.com/questions/26408110/why-does-qhull-error-when-computing-convex-hull-of-a-few-points
# https://gist.github.com/Vini2/2d35132f70ee18298fdea142b5530a52
# https://medium.com/@errazkim/computing-the-convex-hull-in-python-60a6087e0faa
# The expected number of vertices in the convex hull is O(n^(m−1/m+1)) for n→∞. It was proved in 1970 by H. Raynaud in "Sur l’enveloppe convex des nuages de points aleatoires dans Rn".


def load_embeddings(args, dataset_size):
    """Loads the train and test embeddings"""
    with h5py.File(args.node_embeddings, 'r') as f:
        node_embeddings = [np.array(f[f'embedding_{i}']) for i in range(dataset_size)]
    return node_embeddings


def reduce_embedding_dimensionality(embeddings, n_components=2):
    """Reduces the dimensionality of each node embedding"""
    reduced_embeddings = []
    for i in range(len(embeddings)):
        model = PCA(n_components=n_components).fit(embeddings[i])
        reduced_embeddings.append(model.transform(embeddings[i]))
    return reduced_embeddings

def calc_matrix_distances(args):

    dataset = dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    embeddings = load_embeddings(args, len(dataset))

    reduced_embeddings = reduce_embedding_dimensionality(embeddings)

    """
        -> [Cx, Cy, P, V, S, D, Rsh, Rsp]
    """

    convex_hulls = dict.fromkeys(range(len(embeddings)), list()) # centroids
    # convex_hulls = dict.fromkeys(range(len(embeddings)), 0) # volumes or perimeters

    t0 = time()

    for k, _ in convex_hulls.items():

        # convex_hulls[k] = ConvexHullChild(reduced_embeddings[k]).calc_centroid()
        # convex_hulls[k] = ConvexHullChild(reduced_embeddings[k]).calc_perimeter
        # convex_hulls[k] = ConvexHullChild(reduced_embeddings[k]).calc_volume
        # ConvexHullChild(reduced_embeddings[i]).calc_diameter()
        convex_hulls[k] = ConvexHullChild(reduced_embeddings[k]).calc_all_feats()


    convex_hulls_stacked = np.vstack(list(convex_hulls.values()))
    convex_hulls_stacked_normalized = normalize(convex_hulls_stacked, axis=0, norm='max')
    matrix_distances = squareform(pdist(convex_hulls_stacked_normalized, metric='cosine'))
    
    # matrix_distances = squareform(pdist(convex_hulls_stacked, metric='euclidean'))

    # matrix_distances = squareform(pdist([[v] for v in list(convex_hulls.values())], metric='euclidean')) # volumes

    t1 = time()
    computation_time = t1 - t0

    print(computation_time)

    # with open(os.path.join(args.output_dir, 'computation_time.txt'), 'a') as file:
    #     file.write(str(computation_time) + '\n')
    
    # np.save(os.path.join(args.output_dir, f'distances.npy'), matrix_distances)


    






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

    def calc_mom_inertia():
        # distribution of points around centroid
        # up to 2nd moment to capture covariance
        pass
    
    def calc_tunring_functions(self):
        # TODO
        pass

    def calc_all_feats(self):
        centroids = self.calc_centroid()
        area = self.calc_volume()
        perimeter = self.calc_perimeter()
        hull_size = self.calc_hull_size()
        diameter = self.calc_diameter()
        rel_shape = perimeter / area
        rel_spatial = diameter / area
        return [*centroids, area, perimeter, hull_size, diameter, rel_shape, rel_spatial]





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