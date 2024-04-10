import os, pickle, argparse
import numpy as np
from typing import List
from sklearn.metrics import f1_score
from torch_geometric.datasets import TUDataset


def get_nearest_neighbors(M: np.ndarray, 
                          train_idx: List, 
                          test_idx: List) -> dict:
    classification = dict()
    for r in range(M.shape[0]):
        r_idx_sorted = np.argsort(M[r,:])
        classification[test_idx[r]] = [train_idx[x] for x in r_idx_sorted]
    return classification

def get_label_neighbors(dataset: TUDataset, 
                        classification: dict) -> dict:
    label_neighbors = dict()
    for k,v in classification.items():
        label_neighbors[k] = [dataset[x].y.item() for x in v]
    return label_neighbors

def get_ground_truth_labels(dataset: TUDataset,
                            classification: dict) -> List:
    ground_truth = np.array([dataset[x].y.item() for x in classification.keys()])
    return ground_truth

def classify(classification: dict,
             labels: np.ndarray,
             k: int) -> np.ndarray:
    sub_classification = classification[:,:k]
    output = np.vstack([np.sum(sub_classification == x, axis=1) for x in labels])
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
    M = np.load(args.distance_matrix) 
    with open(args.root_indices, 'rb') as fp:
        indices = pickle.load(fp)
    
    train_idx, test_idx = indices[0], indices[1]
    
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    min_k, max_k = 1, M.shape[1] + 1
    
    nearest_neighbors = get_nearest_neighbors(M, train_idx, test_idx)

    label_neighbors = get_label_neighbors(dataset, nearest_neighbors)

    classification = np.vstack(list((label_neighbors.values())))

    label_test_graphs = get_ground_truth_labels(dataset, nearest_neighbors)

    unique_labels = np.unique(classification)
    
    f1scores = []
    for k in range(min_k, max_k):

        output = classify(classification, unique_labels, k)

        f1scores.append(f1_score(label_test_graphs, output, average=args.average))
        
    with open(os.path.join(args.output_dir, 'f1_scores.pkl'), 'wb') as fp:
        pickle.dump(f1scores, fp)
    
    print(max(f1scores))
    print(np.argmax(f1scores)+min_k)
    
    print(np.mean(f1scores)*100)
    print(np.std(f1scores)*100)

    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)