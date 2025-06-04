import unittest
import networkx as nx
from gnnged.calc_edit_cost import EditCost

class TestEditCost(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        print('TestEditCost.__init__')
        super(TestEditCost, self).__init__(*args, **kwargs)

        g1 = [(0, 1), (1, 2)]
        g2 = [(0, 1), (1, 2), (1, 3), (2, 3)]
        source_attrs = {0: 3, 1: 2, 2: 2}
        target_attrs = {0: 1, 1: 2, 2: 1, 3: 3}
        self.source_graph = nx.Graph(g1)
        self.target_graph = nx.Graph(g2)
        nx.set_node_attributes(self.source_graph, source_attrs, 'x')
        nx.set_node_attributes(self.target_graph, target_attrs, 'x')
        self.node_mapping_a = [(0, 1), (1, 0), (2, 2)]
        self.node_mapping_b = [(0, 3), (1, 2), (2, 1)]

    def test_cost_node_edit_a(self):
        edit_cost = EditCost(self.node_mapping_a, self.source_graph, self.target_graph)
        node_cost = edit_cost.compute_cost_node_edit()
        self.assertEqual(node_cost, 4, 'The cost node edit is wrong')
    
    def test_cost_edge_edit_a(self):
        edit_cost = EditCost(self.node_mapping_a, self.source_graph, self.target_graph)
        edge_cost = edit_cost.compute_cost_edge_edit()
        self.assertEqual(edge_cost, 4, 'The cost edge edit is wrong')
    
    def test_cost_node_edit_b(self):
        edit_cost = EditCost(self.node_mapping_b, self.source_graph, self.target_graph)
        node_cost = edit_cost.compute_cost_node_edit()
        self.assertEqual(node_cost, 2, 'The cost node edit is wrong')
    
    def test_cost_edge_edit_b(self):
        edit_cost = EditCost(self.node_mapping_b, self.source_graph, self.target_graph)
        edge_cost = edit_cost.compute_cost_edge_edit()
        self.assertEqual(edge_cost, 2, 'The cost edge edit is wrong')


if __name__ == '__main__':
    unittest.main()