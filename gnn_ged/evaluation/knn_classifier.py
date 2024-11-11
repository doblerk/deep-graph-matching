import os
import json
import argparse
import numpy as np

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from torch_geometric.datasets import TUDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_matrix', type=str, help='Path to distance matrix')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--average', type=str, default='binary', help='Multiclass target')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def knn_classifier(args):

    distance_matrix = np.load(args.distance_matrix)
    
    with open(os.path.join(args.indices_dir, 'train_indices.json'), 'r') as fp:
        train_idx = json.load(fp)
    
    with open(os.path.join(args.indices_dir, 'test_indices.json'), 'r') as fp:
        test_idx = json.load(fp)

    train_distance_matrix = distance_matrix[train_idx,:]
    train_distance_matrix = train_distance_matrix[:, train_idx]
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

    ks = (3, 5, 7, 11)
    best_k = None
    best_score = 0

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        scores = cross_val_score(knn, 
                                 train_distance_matrix,
                                 train_labels,
                                 cv=kf,
                                 scoring=scoring)
        
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    # retrain on the test data with the best k
    knn_test = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed')
    
    knn_test.fit(train_distance_matrix, train_labels)

    predictions = knn_test.predict(test_distance_matrix)

    f1 = f1_score(test_labels, predictions, average=args.average)

    # print(f"F1 Score on new data with K={best_k}: {f1}")
    with open(os.path.join(args.output_dir, 'final_classification_acc.txt'), 'a') as file:
        file.write(f'The optimal number of K-nearest neighbors is {best_k} with a mean F1 score of {best_score}\n')
        file.write(f'F1 Score on new data with K={best_k}: {f1}\n')



def main(args):
    knn_classifier(args)
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)