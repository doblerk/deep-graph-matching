import h5py
import logging
import numpy as np
from sklearn.decomposition import PCA


def load_embeddings(dir, dataset_size):
    """Loads the train and test embeddings"""
    with h5py.File(dir, 'r') as f:
        node_embeddings = {i:np.array(f[f'embedding_{i}']) for i in range(dataset_size)}
    return node_embeddings


def reduce_embedding_dimensionality(embeddings, dims):
    """Reduces the dimensionality of each node embedding"""
    reduced = {}
    for dim in dims:
        reduced[dim] = {}
        for i, embedding in embeddings.items():
            num_nodes, _ = embedding.shape
            if num_nodes >= dim:
                reduced[dim][i] = PCA(n_components=dim).fit_transform(embedding)
            else:
                logging.warning(f"Skipping PCA for dim={dim}, graph={i}.")
                continue
    return reduced


def get_min_graph(dataset):
    """Returns the size of the smaller graph"""
    return min(data.x.shape[0] for data in dataset)