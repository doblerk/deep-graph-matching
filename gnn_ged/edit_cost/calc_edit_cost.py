import numpy as np
import networkx as nx
from typing import List
from scipy.spatial import distance


class EditCost:
    """
    Compute the cost of edit operations implied by the node assignment.

    Attributes
    ----------
    node_mapping : List[List[int]]
        List of node assignments from g1 to g2.
    g1_n : nx.Graph (source graph)
        Source graph.
    g2_n : nx.Graph (target graph)
        Target graph.

    Methods
    -------
    compute_cost_node_edit() -> int
        Computes the cost of node edit operations, considering attributes if specified.
    compute_cost_edge_edit()
        Computes the cost of edge edit operations.
    """

    def __init__(self,
                 node_mapping: List[List[int]],
                 g1_nx: nx.Graph,
                 g2_nx: nx.Graph) -> None:
        
        self.node_mapping = node_mapping
        self.g1_nx = g1_nx
        self.g2_nx = g2_nx
        self.g2_nodes = set(self.g2_nx.nodes)
        self.g2_mapped_nodes = {x[1] for x in self.node_mapping}
        self.g2_unmapped_nodes = self.g2_nodes - self.g2_mapped_nodes

    def compute_cost_node_edit(self, use_attrs: bool = False) -> int:
        """
        Computes the cost of node edit operations.

        Parameters
        ----------
        use_attrs : bool, optional
            Whether to consider node attributes for substitution cost (default is False).

        Returns
        -------
        int
            Total cost of node edit operations.
        """
        cost = 0
        source_attrs = list(nx.get_node_attributes(self.g1_nx, name='x').values())
        target_attrs = list(nx.get_node_attributes(self.g2_nx, name='x').values())
        
        if use_attrs:
            # calculate the cost using Euclidean distance for continuous attributes
            cost += sum(
                distance.euclidean(source_attrs[node1], target_attrs[node2]) 
                for node1, node2 in self.node_mapping
            )
        else:
            # calculate the cost using Dirac distance for one-hot encoded attributes
            cost += sum(
                source_attrs[node1] != target_attrs[node2] 
                for node1, node2 in self.node_mapping
            )
        
        # add cost for unmapped nodes in the target graph
        cost += len(self.g2_unmapped_nodes)
        return cost

    def compute_cost_edge_edit(self) -> int:
        """
        Compute the cost of edge edit operations.

        Returns
        -------
        int
            Total cost of edge edit operations.
        """
        cost = 0
        n = len(self.node_mapping)
        for i in range(n):
            for j in range(i + 1, n):
                
                phi_i = self.node_mapping[i][1]
                phi_j = self.node_mapping[j][1]

                if self.g1_nx.has_edge(i, j):
                    # check for edge substitution or deletion
                    if not self.g2_nx.has_edge(phi_i, phi_j):
                        cost += 1
                elif self.g2_nx.has_edge(phi_i, phi_j):
                    # check for edge insertion
                        cost += 1
    
        cost += sum([self.g2_nx.degree[x] for x in self.g2_unmapped_nodes])
        return cost