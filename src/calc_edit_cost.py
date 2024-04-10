import numpy as np
import networkx as nx
from typing import List

class EditCost:
    """
    Compute the cost of edit operations implied by the node assignment.

    Attributes
    ----------
    node_mapping : List
    g1_n : nx.Graph (source graph)
    g2_n : nx.Graph (target graph)

    Methods
    -------
    compute_cost_node_edit()
    compute_cost_edge_edit()
    """

    def __init__(self,
                 node_mapping: List,
                 g1_nx: nx.Graph,
                 g2_nx: nx.Graph) -> None:
        
        self.node_mapping = node_mapping
        self.g1_nx = g1_nx
        self.g2_nx = g2_nx

        self.g2_nodes = self.g2_nx.nodes
        self.g2_mapped_nodes = sorted([x[1] for x in self.node_mapping])
        self.g2_unmapped_nodes = np.setdiff1d(self.g2_nodes, self.g2_mapped_nodes)

    def compute_cost_node_edit(self):
        cost = 0
        source_attrs = list(nx.get_node_attributes(self.g1_nx, name='x').values())
        target_attrs = list(nx.get_node_attributes(self.g2_nx, name='x').values())
        cost += sum([source_attrs[node1] != target_attrs[node2] for node1, node2 in self.node_mapping])
        cost += len(self.g2_unmapped_nodes)
        return cost

    def compute_cost_edge_edit(self):
        cost = 0
        n = self.g1_nx.number_of_nodes()
        for i in range(n):
            for j in range(i + 1, n):
                
                phi_i = self.node_mapping[i][1]
                phi_j = self.node_mapping[j][1]

                if self.g1_nx.has_edge(i, j):
                    # check for edge substitution
                    if self.g2_nx.has_edge(phi_i, phi_j):
                        pass
                        
                    # check for edge deletion
                    else:
                        cost += 1
                else:
                    if self.g2_nx.has_edge(phi_i, phi_j):
                    # check for edge insertion
                        cost += 1
    
        cost += sum([self.g2_nx.degree[x] for x in self.g2_unmapped_nodes])
        return cost