import os
import argparse

import numpy as np

from torch_geometric.datasets import TUDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):

    # Load the dataset from TUDataset
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    
    # Shuffle the indices
    dataset_idx = np.arange(0, len(dataset), dtype=np.int32)
    np.random.shuffle(dataset_idx)

    # Split the dataset as 80-20 train and test
    train_idx, test_idx = dataset_idx[:int(len(dataset)*0.8)], dataset_idx[int(len(dataset)*0.8):]
    
    # Save the indices
    np.save(os.path.join(args.output_dir, 'train_indices.npy'), train_idx)
    np.save(os.path.join(args.output_dir, 'test_indices.npy'), test_idx)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)