import h5py
import numpy as np
from sklearn.decomposition import PCA


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