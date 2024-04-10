import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment




class NodeMapping:

    def __init__(self,
                 source_embedding: np.ndarray,
                 target_embedding: np.ndarray) -> None:
        
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

    def compute_embedding_distances(self):
        return cdist(self.source_embedding, self.target_embedding, metric='euclidean')

    def compute_node_mapping(self, embedding_distances):
        ri, ci = linear_sum_assignment(embedding_distances)
        return list(zip(ri, ci))