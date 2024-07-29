import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot(pred_distances, true_distances):

    # Load and flatten the matrix distances
    pred_distances_flatten = np.load(pred_distances).flatten(order='C')
    true_distances_flatten = np.load(true_distances).flatten(order='C')

    # Normalize the values
    pred_distances_normalized = ((pred_distances_flatten - np.min(pred_distances_flatten)) / (np.max(pred_distances_flatten) - np.min(pred_distances_flatten))).squeeze()
    true_distances_normalized = ((true_distances_flatten - np.min(true_distances_flatten)) / (np.max(true_distances_flatten) - np.min(true_distances_flatten))).squeeze()
    
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
    parser.add_argument('--distance_matrix_bpged_upper', type=str, help='Path to BP-GED distance matrix')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    plot(args.distance_matrix_pred,
         args.distance_matrix_bpged_upper,
         args.output_dir)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)