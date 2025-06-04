import numpy as np
import networkx as nx
from typing import List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ortools.graph.python import min_cost_flow
from gnnged.assignment import greedy_assignment

class NodeAssignment:
    """
    Compute the node assignment using the source and target embeddings.

    Attributes
    ----------
    source_embedding : np.ndarray
    target_embedding : np.ndarray

    Methods
    -------
    calc_cost_matrix()
    calc_linear_sum_assignment(cost_matrix)
    calc_greedy_assignment(cost_matrix)
    calc_min_cost_flow(cost_matrix)
    """

    def __init__(self,
                 source_embedding: np.ndarray,
                 target_embedding: np.ndarray) -> None:
        
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

    def calc_cost_matrix(self) -> np.ndarray:
        return cdist(self.source_embedding, self.target_embedding, metric='euclidean')

    def calc_linear_sum_assignment(self, 
                                   cost_matrix : np.ndarray) -> List[List[int]]:
        ri, ci = linear_sum_assignment(cost_matrix)
        return list(zip(ri, ci))
    
    def calc_greedy_assignment(self,
                               cost_matrix : np.ndarray) -> List[List[int]]:
        ci = greedy_assignment.calc_greedy_assignment_fast(cost_matrix)
        return list(zip(range(len(ci)), ci))

    def calc_min_cost_flow(self, cost_matrix):

        smcf = min_cost_flow.SimpleMinCostFlow()

        cost_matrix = (cost_matrix * 100).astype(np.uint32)

        num_workers, num_tasks = cost_matrix.shape

        source = num_workers + num_tasks
        sink = source + 1

        # add edges from the source to each worker
        for worker in range(num_workers):
            smcf.add_arc_with_capacity_and_unit_cost(
                source, worker, 1, 0,
            )
        
        # add edges from each task to the sink
        for task in range(num_tasks):
            smcf.add_arc_with_capacity_and_unit_cost(
                num_workers + task, sink, 1, 0,
            )
        
        # add edges between workers and tasks
        for worker in range(num_workers):
            for task in range(num_tasks):
                smcf.add_arc_with_capacity_and_unit_cost(
                    worker, num_workers + task, 1, cost_matrix[worker][task],
                )
        
        # set the supply and demand
        smcf.set_node_supply(source, num_workers)
        smcf.set_node_supply(sink, -num_workers)

        status = smcf.solve()

        assignment = []
        for arc in range(smcf.num_arcs()):
            if smcf.tail(arc) != source and smcf.head(arc) != sink:
                if smcf.flow(arc) > 0:
                    assignment.append((smcf.tail(arc), smcf.head(arc) - num_workers))
        
        return assignment