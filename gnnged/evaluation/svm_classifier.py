import os
import json
import argparse
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from torch_geometric.datasets import TUDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_matrix', type=str, help='Path to distance matrix')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--average', type=str, default='binary', help='Multiclass target')
    return parser


def svm_classifier(args):
    
    distance_matrix = np.load(args.distance_matrix)

    train_idx = sorted(np.load(os.path.join(args.indices_dir, 'train_indices.npy')))
    test_idx = sorted(np.load(os.path.join(args.indices_dir, 'test_indices.npy')))

    idx = list(range(distance_matrix.shape[0]))

    # slicing with np.setdiff1d returns sorted idx -> same order as distance matrix
    train_distance_matrix = distance_matrix[np.setdiff1d(idx, test_idx),:]
    train_distance_matrix = train_distance_matrix[:, np.setdiff1d(idx, test_idx)]
    np.fill_diagonal(train_distance_matrix, 1000)

    test_distance_matrix = distance_matrix[test_idx,:]
    test_distance_matrix = test_distance_matrix[:,train_idx]

    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    
    train_labels = list(dataset[train_idx].y.numpy())
    test_labels = list(dataset[test_idx].y.numpy())

    if args.average == 'average':
        scoring = 'f1'
    else:
        scoring = 'f1_micro'
    
    C = 1.0
    kernel = 'precomputed'

    svm_test = SVC(C=C, kernel=kernel)
    svm_test.fit(train_distance_matrix, train_labels)

    predictions = svm_test.predict(test_distance_matrix)

    f1 = f1_score(test_labels, predictions, average=args.average)

    print(f"F1 Score on new data (SVM) {f1}")


def main(args):
    svm_classifier(args)
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)