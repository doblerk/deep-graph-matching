import os, pickle, argparse
import numpy as np
from typing import List
from sklearn.metrics import f1_score
from torch_geometric.datasets import TUDataset


class ClassifyGraphs:
    """
    Classify each test graph using k-nearest neighbors with a majority voting scheme.

    Attributes
    ----------
    D : np.ndarray
    dataset : TUDataset

    Methods
    -------
    get_nearest_graphs(train_idx, test_idx)
    get_label_nearest_graphs(nearest_graphs)
    get_ground_truth_label(nearest_graphs)
    classify(nearest_graphs, ground_truth_label, k)
    """

    def __init__(self,
                 D: np.ndarray,
                 dataset: TUDataset) -> None:
        self.D = D
        self.dataset = dataset

    def get_nearest_graphs(self,
                           train_idx: List[int],
                           test_idx: List[int]) -> dict[int, List[int]]:
        nearest_graphs = dict()
        for r in range(self.D.shape[0]):
            r_idx_sorted = np.argsort(self.D[r,:])
            nearest_graphs[test_idx[r]] = [train_idx[x] for x in r_idx_sorted]
        return nearest_graphs

    def get_label_nearest_graphs(self,
                                 nearest_graphs: dict[int, List[int]]) -> dict[int, int]:
        laebel_nearest_graphs = dict()
        for k,v in nearest_graphs.items():
            laebel_nearest_graphs[k] = [self.dataset[x].y.item() for x in v]
        return laebel_nearest_graphs
    
    def get_ground_truth_label(self,
                               nearest_graphs: dict[int, List[int]]) -> np.ndarray:
        ground_truth_label = np.array([self.dataset[x].y.item() for x in nearest_graphs.keys()])
        return ground_truth_label
    
    def classify(self,
                 nearest_graphs: dict[int, List[int]],
                 ground_truth_label: np.ndarray,
                 k: int) -> np.ndarray:
        sub_classification = nearest_graphs[:,:k]
        output = np.vstack([np.sum(sub_classification == x, axis=1) for x in ground_truth_label])
        majority_vote = np.argmax(output, axis=0)
        return majority_vote



def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--distance_matrix', type=str, help='Path to distance matrix')
    parser.add_argument('--root_indices', type=str, help='Path to indices')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--average', type=str, default='binary', help='Multiclass target')
    return parser


def main(args):
    D = np.load(args.distance_matrix)
    with open(args.root_indices, 'rb') as fp:
        indices = pickle.load(fp)
    
    train_idx, test_idx = indices[0], indices[1]
    
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    min_k, max_k = 1, D.shape[1] + 1

    classification = ClassifyGraphs(D, dataset)
    
    nearest_graphs = classification.get_nearest_graphs(D, train_idx, test_idx)

    label_nearest_graphs = classification.get_label_nearest_graphs(dataset, nearest_graphs)

    label_nearest_graphs_stacked = np.vstack(list((label_nearest_graphs.values())))

    label_test_graphs = classification.get_ground_truth_label(dataset, nearest_graphs)

    unique_labels = np.unique(label_nearest_graphs_stacked)
    
    f1scores = []
    for k in range(min_k, max_k):

        output = classification.classify(nearest_graphs, unique_labels, k)

        f1scores.append(f1_score(label_test_graphs, output, average=args.average))
        
    with open(os.path.join(args.output_dir, 'f1_scores.pkl'), 'wb') as fp:
        pickle.dump(f1scores, fp)

    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)