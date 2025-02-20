import os, h5py, json, argparse
import numpy as np
from time import time
from random import choices, seed
from itertools import chain, repeat
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist
from torch_geometric.datasets import TUDataset


def load_indices(idx_path):
    '''Loads indices from a JSON file'''
    with open(idx_path, 'r') as fp:
        return json.load(fp)

def load_embeddings(embeddings_path, dataset):
    '''Loads node embeddings from an HDF5 file'''
    with h5py.File(embeddings_path, 'r') as f:
        return {i:np.array(f[f'embedding_{i}']) for i in range(len(dataset))}

def get_indices(idx, dataset):
    '''Generates training and testing indices based on data set node counts'''
    return np.array(list(chain.from_iterable(repeat(idx, dataset[idx].num_nodes) for idx in idx)))

def get_labels(test_idx, testing_indices, training_indices, dataset, distances):
    '''Predicts labels for test data and collects true labels'''
    predicted_labels = []
    true_labels = []

    for idx in test_idx:
        indices = np.where(testing_indices == idx)[0]
        sub_distances = distances[indices, :]
        nearest_neighbors = np.argmin(sub_distances, axis=1)

        nearest_labels = [dataset[training_indices[neighbor]].y.item() for neighbor in nearest_neighbors]
        predicted_label = np.argmax(np.bincount(nearest_labels))
        predicted_labels.append(predicted_label)

        true_labels.append(dataset[idx].y.item())
    
    return predicted_labels, true_labels

def get_labels_knn(test_idx, testing_indices, training_indices, dataset, distances, k=1):
    '''Predicts labels for test data using k-nearest neighbor and collects true labels'''
    graph_predictions = []

    for idx in test_idx:
        node_indices = np.where(testing_indices == idx)[0]
        sub_distances = distances[node_indices, :]

        node_labels = []
        for node_distances in sub_distances:
            nearest_neighbors = np.argsort(node_distances)[:k]
            nearest_labels = [dataset[training_indices[neighbor]].y.item() for neighbor in nearest_neighbors]
            node_label = np.argmax(np.bincount(nearest_labels))
            node_labels.append(node_label)
        
        graph_label = np.argmax(np.bincount(node_labels))
        graph_predictions.append(graph_label)
        
    true_labels = [dataset[idx].y.item() for idx in test_idx]
    
    return graph_predictions, true_labels


def get_labels_weighted_knn(test_idx, testing_indices, training_indices, dataset, distances, k=1):
    '''Predicts labels for test data using k-nearest neighbors with softmax-weighted probabilities and collects true labels'''
    graph_predictions = []

    epsilon = 1e-12

    for idx in test_idx:
        node_indices = np.where(testing_indices == idx)[0]
        sub_distances = distances[node_indices, :]

        node_labels = []
        for node_distances in sub_distances:
            nearest_neighbors = np.argsort(node_distances)[:k]
            nearest_distances = node_distances[nearest_neighbors]

            softmax_weights = np.exp(-nearest_distances) / np.sum(np.exp(-nearest_distances))

            class_probabilities = {}
            for neighbor, weight in zip(nearest_neighbors, softmax_weights):
                neighbor_label = dataset[training_indices[neighbor]].y.item()
                class_probabilities[neighbor_label] = class_probabilities.get(neighbor_label, 0) + np.log(max(weight, epsilon))
            
            predicted_label = max(class_probabilities, key=class_probabilities.get)
            node_labels.append(predicted_label)
        
        graph_label = np.argmax(np.bincount(node_labels))
        graph_predictions.append(graph_label)

    true_labels = [dataset[idx].y.item() for idx in test_idx]
    
    return graph_predictions, true_labels

def get_accuracy(predicted_labels, true_labels):
    '''Calculates the F1 score'''
    return f1_score(true_labels, predicted_labels)

def create_bags(train_idx, num_bags=50):
    '''Creates bags by randomly sampling training graphs with replacement'''
    seed(30)
    num_graphs = len(train_idx)
    return [sorted(choices(train_idx, k=num_graphs)) for _ in range(num_bags)]

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

    training_indices = get_indices(train_idx, dataset)
    testing_indices = get_indices(test_idx, dataset)

    predicted_labels, true_labels = get_labels_weighted_knn(
        test_idx, testing_indices, training_indices, dataset, distances, k=5
    )

    accuracy = get_accuracy(predicted_labels, true_labels)
    print(f"Classification Accuracy (F1): {accuracy:.4f}")

    # ENSEMBLE
    # predicted_labels = []
    # true_labels = []
    # for bag in bags:
    #     training_embeddings = np.vstack([node_embeddings[i] for i in bag])
    #     training_indices = list(chain.from_iterable(repeat(idx, dataset[idx].num_nodes) for idx in bag))

    #     distances = cdist(testing_embeddings, training_embeddings)        

    #     pred_labels, tr_labels = get_labels(
    #         test_idx, testing_indices, training_indices, dataset, distances
    #     )
    #     predicted_labels.append(pred_labels)
    #     true_labels.append(tr_labels)

    # aggregated_predictions = []
    # for i in range(len(test_idx)):
    #     # collect all predictions for the i-th graph from all bags
    #     graph_predictions = [predicted_labels[bag_idx][i] for bag_idx in range(len(bags))]
    #     # majority voting
    #     aggregated_label = np.argmax(np.bincount(graph_predictions))
    #     aggregated_predictions.append(aggregated_label)
    # true_graph_labels = [dataset[idx].y.item() for idx in test_idx]

    # accuracy = get_accuracy(predicted_labels, true_labels)
    # accuracy = get_accuracy(aggregated_predictions, true_graph_labels)

    # with open(os.path.join(args.output_dir, 'results_lsap.txt'), 'a') as file:
    #     file.write(f'Computation time: {computation_time}\n')
    #     file.write(f'Classification accuracy (F1): {accuracy}\n')


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)