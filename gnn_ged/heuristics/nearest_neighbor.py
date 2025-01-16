import os, h5py, json, argparse
import numpy as np
from time import time
from itertools import chain, repeat
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from torch_geometric.datasets import TUDataset


def load_indices(idx_path):
    '''Loads indices from a JSON file'''
    with open(idx_path, 'r') as fp:
        return json.load(fp)

def load_embeddings(embeddings_path, dataset):
    '''Loads node embeddings from an HDF5 file'''
    with h5py.File(embeddings_path, 'r') as f:
        return {i:np.array(f[f'embedding_{i}']) for i in range(len(dataset))}

def get_indices(train_idx, test_idx, dataset):
    '''Generates training and testing indices based on data set node counts'''
    training_indices = list(chain.from_iterable(repeat(idx, dataset[idx].num_nodes) for idx in train_idx))
    testing_indices = list(chain.from_iterable(repeat(idx, dataset[idx].num_nodes) for idx in test_idx))
    return np.array(training_indices), np.array(testing_indices)

def get_labels(test_idx, testing_indices, testing_embeddings, training_indices, training_embeddings, dataset, distances):
    '''Predicts labels for test data and collects true labels'''
    predicted_labels = []
    true_labels = []

    for idx in test_idx:
        indices = np.where(testing_indices == idx)[0]
        sub_distances = distances[indices, :]
        nearest_neighbors = np.argmin(sub_distances, axis=1)
        # _, nearest_neighbors = linear_sum_assignment(sub_distances)

        nearest_labels = [dataset[training_indices[neighbor]].y.item() for neighbor in nearest_neighbors]
        predicted_label = np.argmax(np.bincount(nearest_labels))
        predicted_labels.append(predicted_label)

        true_labels.append(dataset[idx].y.item())
    
    return predicted_labels, true_labels

def get_accuracy(predicted_labels, true_labels):
    '''Calculates the F1 score'''
    return f1_score(true_labels, predicted_labels)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--node_embeddings', type=str, help='Path to node embeddings file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser

def main(args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    train_idx = load_indices(os.path.join(args.indices_dir, 'train_indices.json'))
    test_idx = load_indices(os.path.join(args.indices_dir, 'test_indices.json'))

    node_embeddings = load_embeddings(args.node_embeddings, dataset)

    training_embeddings = np.vstack([node_embeddings[i] for i in train_idx])
    testing_embeddings = np.vstack([node_embeddings[i] for i in test_idx])

    distances = cdist(testing_embeddings, training_embeddings)

    training_indices, testing_indices = get_indices(train_idx, test_idx, dataset)

    t0 = time()
    predicted_labels, true_labels = get_labels(
        test_idx, testing_indices, testing_embeddings, training_indices, training_embeddings, dataset, distances
    )
    t1 = time()
    computation_time = t1 - t0

    accuracy = get_accuracy(predicted_labels, true_labels)

    with open(os.path.join(args.output_dir, 'results_lsap.txt'), 'a') as file:
        file.write(f'Computation time: {computation_time}\n')
        file.write(f'Classification accuracy (F1): {accuracy}\n')


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)