import os
import argparse
import numpy as np
from time import time
from grakel import GraphKernel
from grakel.datasets import fetch_dataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--kernel', type=str, choices=['sp', 'rw', 'wl'], help='Graph kernel')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):

    dataset = fetch_dataset(name=args.dataset_name, 
                            verbose=True,
                            data_home=args.dataset_dir,
                            produce_labels_nodes=True,
                            as_graphs=False)
    print(dataset.data[0])
    # match args.kernel:

    #     case 'sp':
    #         kernel = GraphKernel(kernel={'name':'shortest_path', 'algorithm_type':'floyd_warshall'}, normalize=True)
        
    #     case 'rw':
    #         kernel = GraphKernel(kernel={'name':'random_walk'}, normalize=True)
        
    #     case 'wl':
    #         kernel = GraphKernel(kernel={'name':'weisfeiler_lehman', 'n_iter':5}, normalize=True)

    # # t0 = time()
    # similarity_matrix = kernel.fit_transform(dataset.data)
    # dissimilarity_matrix = np.max(similarity_matrix) - similarity_matrix
    # # t1 = time()
    # # computation_time = t1 - t0
    # print(similarity_matrix)

    # dissimilarity_matrix[dissimilarity_matrix < 0.0] = 0.0
    # dissimilarity_matrix[dissimilarity_matrix == 'NaN'] = 1.0

    # with open(os.path.join(args.output_dir, f'computation_time_kernels.txt'), 'a') as file:
    #     file.write(f'Computation time for {args.kernel} kernel: {computation_time}\n')

    # np.save(os.path.join(args.output_dir, f'{args.kernel}_distances.npy'), dissimilarity_matrix)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)