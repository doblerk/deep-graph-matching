import os
import pickle
import argparse

import numpy as np

from time import time
from sklearn.decomposition import PCA
from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform


# https://stackoverflow.com/questions/26408110/why-does-qhull-error-when-computing-convex-hull-of-a-few-points
# https://gist.github.com/Vini2/2d35132f70ee18298fdea142b5530a52
# https://medium.com/@errazkim/computing-the-convex-hull-in-python-60a6087e0faa
# The expected number of vertices in the convex hull is O(n^(m−1/m+1)) for n→∞. It was proved in 1970 by H. Raynaud in "Sur l’enveloppe convex des nuages de points aleatoires dans Rn".


def load_dataset(args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    dataset_nx = [to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))]
    return dataset_nx


def load_embeddings(args):
    with open(os.path.join(args.output_dir, 'train_embeddings.pkl'), 'rb') as fp:
        train_embeddings = pickle.load(fp)
    
    with open(os.path.join(args.output_dir, 'test_embeddings.pkl'), 'rb') as fp:
        test_embeddings = pickle.load(fp)
    
    return train_embeddings, test_embeddings


def reduce_dimensionality(embeddings, n_components=2):
    model = PCA(n_components=n_components).fit(embeddings)
    return model.transform(embeddings)


# add child class for implementing something additional like centroids
class ConvexHullChild(ConvexHull):

    def __init__(self, points):
        ConvexHull.__init__(self, points, qhull_options='QJ')
        self.centroids = self.calc_centroids()

    def calc_centroids(self):
        c = []
        for i in range(self.points.shape[1]):
            c.append(np.mean(self.points[self.vertices,i]))
        return c


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    
    dataset_nx = load_dataset(args)

    train_embeddings, test_embeddings = load_embeddings(args)
   
    embeddings = dict()
    for i in range(len(dataset_nx)):
        if i in train_embeddings:
            # if train_embeddings[i].shape[0] >= 10:
            embeddings[i] = reduce_dimensionality(train_embeddings[i], n_components=2)
        else:
            # if test_embeddings[i].shape[0] >= 10:
            embeddings[i] = reduce_dimensionality(test_embeddings[i], n_components=2)

    

    # what takes quite some time is the linear projection using PCA. We could bypass this by either doing it offline as a preprocessing step or by outputting smaller dimensions with the GNNs
    # with GNNs I cannot produce n_dim specific for each graph, but I can choose n_dim based on the smaller graph?

    # g1 = train_embeddings[139]
    # g2 = train_embeddings[133]

    # proj_points1 = reduce_dimensionality(g1, n_components=2)
    # convex_hull1 = ConvexHullChild(proj_points1, qhull_options='QJ')

    # proj_points2 = reduce_dimensionality(g2, n_components=2)
    # convex_hull2 = ConvexHullChild(proj_points2, qhull_options='QJ')
    
    # case 1: using the volume
    # vol1 = convex_hull1.volume
    # vol2 = convex_hull2.volume
    # then compare these volums using different dissimilarity functions

    # case 2: using the centroid
    # c1 = convex_hull1.centroid()
    # c2 = convex_hull2.centroid()
    # then compare these distances using euclidean distance

    """
        TODO:
                - Compute convex hulls for each graph and associated values like volumes, centroids etc.
                - Compare these values between graphs to define the dissimilarity.
                - Record runtimes and accuracies.
                - Then, think of alternative ways to compare these hulls
    """

    # 1. reduce the dimensionality of each node embedding
    # should we do it offline or should we choose a smaller hidden dimension with GNNs?
    # done in the first step

    # points = np.random.rand(12, 2)
    # t0 = time()
    # for i in range(188):
    #     ConvexHull(points)
    # t1 = time()
    # print(t1-t0)

    # 2. compute convex hulls for each graph
    # distances are float... should they be int or scaled?
    convex_hulls = dict.fromkeys(range(len(dataset_nx)), list()) # centroids
    convex_hulls = dict.fromkeys(range(len(dataset_nx)), 0) # volumes
    convex_hulls = dict.fromkeys(embeddings.keys(), list()) # for enzymes 
    
    t0 = time()
    for k, v in convex_hulls.items():
        # convex_hulls[k] = ConvexHullChild(embeddings[k]).centroids
        convex_hulls[k] = ConvexHullChild(embeddings[k]).volume

    # convex_hulls_stacked = np.vstack(list(convex_hulls.values())) # centroids
    # matrix_distances = squareform(pdist(convex_hulls_stacked)) # centroids
    matrix_distances = squareform(pdist([[v] for v in list(convex_hulls.values())], metric='euclidean')) # volumes
    t1 = time()
    print(t1-t0)
    
    # np.save(os.path.join(args.output_dir, 'heuristic/distances_volumes.npy'), matrix_distances)
    print([[v] for v in list(convex_hulls.values())])
    print(len([[v] for v in list(convex_hulls.values())]))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)