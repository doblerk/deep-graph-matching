import numpy as np
from typing import List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class NodeAssignment:
    """
    Compute the node assignment using the source and target embeddings.

    Attributes
    ----------
    source_embedding : np.ndarray
    target_embedding : np.ndarray

    Methods
    -------
    compute_embedding_distances()
    compute_node_assignment(embedding_distances)
    """

    def __init__(self,
                 source_embedding: np.ndarray,
                 target_embedding: np.ndarray) -> None:
        
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

    def compute_embedding_distances(self) -> np.ndarray:
        return cdist(self.source_embedding, self.target_embedding, metric='euclidean')

    def compute_node_assignment(self, 
                             embedding_distances: np.ndarray) -> List[List[int]]:
        ri, ci = linear_sum_assignment(embedding_distances)
        return list(zip(ri, ci))