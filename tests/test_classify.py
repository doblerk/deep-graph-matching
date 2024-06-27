import unittest
import numpy as np
from gnn_ged.classify import ClassifyGraphs

class TestClassifyGraphs(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        print('TestClassifyGraphs.__init__')
        super(TestClassifyGraphs, self).__init__(*args, **kwargs)

        # Matrix of distances: test (4) vs train (6) graphs
        self.matrix_distances = np.array([
            [1, 2, 3, 4, 5, 6],
            [4, 6, 2, 9, 12, 8],
            [8, 3, 9, 14, 7, 1],
            [7, 8, 9, 10, 11, 12],
            ],
        )

        self.test_idx = [0, 1, 2, 3]
        self.train_idx = [4, 5, 6, 7, 8, 9]
        
        self.dataset = None

        self.nearest_graphs = {0: [4,5,6,7,8,9],
                               1: [6,4,5,9,7,8],
                               2: [9,5,8,4,6,7],
                               3: [4,5,6,7,8,9]}
        
        self.label_nearest_graphs = {0: [0,0,0,1,1,1],
                                     1: [0,0,0,1,1,1],
                                     2: [1,0,1,0,0,1],
                                     3: [0,0,0,1,1,1]}
        
        self.label_test_graphs = [0, 0, 1, 0]
    

    def test_nearest_graphs(self):
        classification = ClassifyGraphs(self.matrix_distances, self.dataset)
        nearest_graphs = classification.get_nearest_graphs(self.train_idx, self.test_idx)
        self.assertListEqual(list(nearest_graphs.values()), list(self.nearest_graphs.values()), 'Nearest graphs are wrong')
    
 
    def test_classification(self):
        classification = ClassifyGraphs(self.matrix_distances, self.dataset)
        label_nearest_graphs_stacked = np.vstack(list((self.label_nearest_graphs.values())))
        unique_labels = np.unique(label_nearest_graphs_stacked)
        output = list(classification.classify(label_nearest_graphs_stacked, unique_labels, 3))
        self.assertListEqual(output, self.label_test_graphs, 'Classification is wrong')


if __name__ == '__main__':
    unittest.main()