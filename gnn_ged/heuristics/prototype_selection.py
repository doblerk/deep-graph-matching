import os
import pickle
import argparse

import numpy as np

from time import time
from collections import defaultdict
from itertools import combinations, product
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset

from scipy.spatial.distance import cdist, pdist, squareform

from gnn_ged.edit_cost.calc_edit_cost import EditCost


def load_dataset(args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    dataset_nx = [to_networkx(dataset[i], node_attrs='x', to_undirected=True) for i in range(len(dataset))]
    return dataset_nx


def load_graph_embeddings(args):
    with open(os.path.join(args.output_dir, 'graph_train_embeddings.pkl'), 'rb') as fp:
        train_embeddings = pickle.load(fp)
    
    with open(os.path.join(args.output_dir, 'graph_test_embeddings.pkl'), 'rb') as fp:
        test_embeddings = pickle.load(fp)
    
    return train_embeddings, test_embeddings

def load_node_embeddings(args):
    with open(os.path.join(args.output_dir, 'train_embeddings.pkl'), 'rb') as fp:
        train_embeddings = pickle.load(fp)
    
    with open(os.path.join(args.output_dir, 'test_embeddings.pkl'), 'rb') as fp:
        test_embeddings = pickle.load(fp)
    
    return train_embeddings, test_embeddings


def calc_spanning_prototype_selector(graph_embeddings, num_prototypes):
    '''
    Selects a set of prototype graphs that span the set of graph embeddings

    Parameters
    ----------
    graph_embeddings : np.array
        2D array of graph embeddings
    num_prototypes : int
        Number of prototypes to select

    Returns
    -------
    list
        List of selected prototype indices
    '''

    if len(graph_embeddings) <= num_prototypes:
        return list(range(0, len(graph_embeddings)))

    P = list()
    T = list(range(graph_embeddings.shape[0]))

    # if distance matrix precomputed
    graph_embeddings = squareform(pdist(graph_embeddings, metric='euclidean'))
    row_sum = np.sum(graph_embeddings, axis=1)
    median_graph_idx = np.argmin(row_sum)

    # or if we have graph embeddings
    # median_graph_idx = calc_set_median_graph(graph_embeddings)

    P.append(median_graph_idx)
    T.remove(median_graph_idx)

    while len(P) < num_prototypes:
        
        # extract submatrix of graph embeddings corresponding to selected prototypes
        prototype_distances = graph_embeddings[P, :]

        # zero out distances between selected prototypes
        prototype_distances[:, P] = 0

        # if we only have one prototype (at the start), use its distances directly
        if len(P) == 1:
            # [[0 1 2 3]] -> [[0 1 2 3]]
            min_distances = prototype_distances
        else:
            # [[0 1 2 3]
            #  [4 5 6 7]] -> [[0 1 2 3]]
            min_distances = np.min(prototype_distances, axis=0)
        
        # select the point that maximizes the minimum distance to the prototypes
        # [[0 1 2 3]] -> 3
        next_prototype_idx = np.argmax(min_distances)
        
        # update the lists
        P.append(next_prototype_idx)
        T.remove(next_prototype_idx)

    return P


def calc_groups(train_idx, dataset):
    '''Calculates the number of nodes for each graph and the unique of unique values'''
    groups = np.array([dataset[x].number_of_nodes() for x in train_idx])
    # groups = np.array([v.shape[0] for v in embeddings.values()])
    unique_groups = np.unique(groups).tolist()
    return groups, unique_groups
   

def calc_graph_groups(groups, unique_groups):
    '''Retrieves the indices of each graph per group'''
    return [list(np.where(groups == grp)[0]) for grp in unique_groups]


def get_prototypes(unique_groups, graph_groups, graph_embeddings, indices, num_prototypes):
    '''Retrieves the prototype graphs for each group'''
    prototypes = {}
    for group_size, group_indices in zip(unique_groups, graph_groups):

        embeddings = np.vstack([graph_embeddings[indices[idx]] for idx in group_indices])

        idx = calc_spanning_prototype_selector(embeddings, num_prototypes)
        
        prototypes[group_size] = [group_indices[i] for i in idx]

    return prototypes


def construct_distance_matrix():
    pass


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    
    dataset_nx = load_dataset(args)

    graph_train_embeddings, graph_test_embeddings = load_graph_embeddings(args)

    train_embeddings, test_embeddings = load_node_embeddings(args)

    train_idx, test_idx = list(graph_train_embeddings.keys()), list(graph_test_embeddings.keys())
    
    embeddings = dict()
    for i in range(len(dataset_nx)):
        if i in graph_train_embeddings:
            embeddings[i] = graph_train_embeddings[i]
        else:
            embeddings[i] = graph_test_embeddings[i]
    
    dic_number_of_nodes = {}
    for i in range(len(dataset_nx)):
        dic_number_of_nodes[i] = dataset_nx[i].number_of_nodes()

    n_train_graphs, n_test_graphs = len(graph_train_embeddings), len(graph_test_embeddings)

    matrix_distances = np.zeros(shape=(n_train_graphs + n_test_graphs, n_train_graphs + n_test_graphs), dtype=np.int32)

    # New from here
    groups, unique_groups = calc_groups(train_idx, dataset_nx)
    # groups -> each number is the number of nodes associated with train_idx
    # unique_groups -> sorted number of unique number of groups

    graph_groups = calc_graph_groups(groups, unique_groups)

    # print(max(map(len, graph_groups)))
    number_unique_prototypes = np.unique([len(x) for x in graph_groups])

    # num_prototypes = 15
    total_prototypes = []
    runtimes = []
    for num_prototypes in number_unique_prototypes:

        prototypes = get_prototypes(unique_groups, graph_groups, graph_train_embeddings, train_idx, num_prototypes)

        # generate all pairwise combinations
        flattened_list = [item for sublist in prototypes.values() for item in sublist]
        # print('Number of prototypes in total: ', len(flattened_list))
        total_prototypes.append(len(flattened_list))
        pairwise_combinations = list(combinations(flattened_list, 2))

        pairwise_self_combinations = []
        for group in prototypes.values():
            tmp = list(combinations(group, 2))
            pairwise_self_combinations.extend(tmp)
        
        unique_pairwise_combinations = [x for x in pairwise_combinations if x not in pairwise_self_combinations]

        # run
        costs = defaultdict(list)
        t0 = time()
        for g, h in unique_pairwise_combinations:

            distances = cdist(train_embeddings[train_idx[g]], train_embeddings[train_idx[h]], metric='euclidean')

            ri, ci = linear_sum_assignment(distances)
            
            edit_cost = EditCost(list(zip(ri, ci)), dataset_nx[train_idx[g]], dataset_nx[train_idx[h]])

            costs[(dic_number_of_nodes[train_idx[g]], dic_number_of_nodes[train_idx[h]])].append(edit_cost.compute_cost_node_edit() + edit_cost.compute_cost_edge_edit())

        # compute mean for each cost, if applicable
        costs_mean = {key: np.mean(val) for key, val in costs.items()}

        # construct matrix
        for x, y in combinations(np.arange(0, matrix_distances.shape[0]), 2):
            g_num_nodes = dic_number_of_nodes[x]
            h_num_nodes = dic_number_of_nodes[y]

            if (g_num_nodes, h_num_nodes) in costs_mean:
                matrix_distances[x,y] = costs_mean[(g_num_nodes, h_num_nodes)]
            elif (h_num_nodes, g_num_nodes) in costs_mean:
                matrix_distances[x,y] = costs_mean[(h_num_nodes, g_num_nodes)]
            else:
                matrix_distances[x,y] = 0
        
        matrix_distances += matrix_distances.T

        t1 = time()
        # print(t1-t0)
        runtimes.append(t1-t0)
        
        np.save(os.path.join(args.output_dir, f'heuristic/distances_{num_prototypes}.npy'), matrix_distances)

    print(number_unique_prototypes)
    print(total_prototypes)
    print(runtimes)






if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)