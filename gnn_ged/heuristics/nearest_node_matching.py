import os, h5py, json, argparse
import numpy as np
from time import time
from itertools import repeat
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist
from torch_geometric.datasets import TUDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--node_embeddings', type=str, help='Path to node embeddings file')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    with open(os.path.join(args.indices_dir, 'train_indices.json'), 'r') as fp:
        train_idx = json.load(fp)
    
    with open(os.path.join(args.indices_dir, 'test_indices.json'), 'r') as fp:
        test_idx = json.load(fp)

    with h5py.File(args.node_embeddings, 'r') as f:
            node_embeddings = {i:np.array(f[f'embedding_{i}']) for i in range(len(dataset))}
    
    training_embeddings = np.vstack([node_embeddings[i] for i in train_idx])
    testing_embeddings = np.vstack([node_embeddings[i] for i in test_idx])

    training_indices, testing_indices = [], []

    for i in range(len(train_idx)):
        training_indices.extend(list(repeat(train_idx[i], dataset[train_idx[i]].num_nodes)))

    for i in range(len(test_idx)):
        testing_indices.extend(list(repeat(test_idx[i], dataset[test_idx[i]].num_nodes)))

    training_indices, testing_indices = np.array(training_indices), np.array(testing_indices)

    predicted_labels, true_labels = [], []

    t0 = time()

    for idx in test_idx:
        indices = np.where(testing_indices == idx)[0]
        distances = cdist(testing_embeddings[indices], training_embeddings)
        nearest_neighbors = np.argmin(distances, axis=1)
        nearest_neighbors_labels = [dataset[training_indices[nghbr]].y.item() for nghbr in nearest_neighbors]
        predicted_labels.append(np.argmax([nearest_neighbors_labels.count(0), nearest_neighbors_labels.count(1)])) # binary target
        true_labels.append(dataset[idx].y.item())
    
    t1 = time()
    computation_time = t1 - t0
    
    accuracy = f1_score(true_labels, predicted_labels)

    with open(os.path.join(args.output_dir, 'classification_1nn.txt'), 'a') as file:
        file.write(f'Computation time: {computation_time} and F1 Score on new data: {accuracy}\n')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)