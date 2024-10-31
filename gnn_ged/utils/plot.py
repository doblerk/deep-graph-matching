import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot(args):

    with open(os.path.join(args.indices_dir, 'train_indices.json'), 'r') as fp:
        train_idx = json.load(fp)
    
    with open(os.path.join(args.indices_dir, 'test_indices.json'), 'r') as fp:
        test_idx = json.load(fp)

    # Load and flatten the matrix distances
    pred_distances = np.load(args.distance_matrix_pred)
    true_distances = np.load(args.distance_matrix_true)
    
    # Process the matrices to get subsamples
    test_pred_distance_matrix = pred_distances[test_idx, :]
    test_pred_distance_matrix = test_pred_distance_matrix[:, train_idx]
    test_pred_distance_matrix = test_pred_distance_matrix.flatten(order='C')

    test_true_distance_matrix = true_distances[test_idx, :]
    test_true_distance_matrix = test_true_distance_matrix[:, train_idx]
    test_true_distance_matrix = test_true_distance_matrix.flatten(order='C')


    # Normalize the values
    pred_distances_normalized = ((test_pred_distance_matrix - np.min(test_pred_distance_matrix)) / (np.max(test_pred_distance_matrix) - np.min(test_pred_distance_matrix))).squeeze()
    true_distances_normalized = ((test_true_distance_matrix - np.min(test_true_distance_matrix)) / (np.max(test_true_distance_matrix) - np.min(test_true_distance_matrix))).squeeze()
    
    # Plot Pred vs True distances
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(true_distances_normalized, pred_distances_normalized, s=26, alpha=0.75, edgecolor='lightgrey', c='grey', linewidth=0.5)
    ax.plot(true_distances_normalized, true_distances_normalized, linestyle='-', linewidth=1, color='black')
    ax.set_xlabel('BP-GED', fontsize=22)
    ax.set_ylabel('GNN-GED', fontsize=22)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'scatter_plot.png'), dpi=250)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_matrix_pred', type=str, help='Path to predicted distance matrix')
    parser.add_argument('--distance_matrix_true', type=str, help='Path to BP-GED distance matrix')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    plot(args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)