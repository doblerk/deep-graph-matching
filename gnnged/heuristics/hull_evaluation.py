import sys
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from torch_geometric.datasets import TUDataset


def knn_classifier(config, output_dir, dim):
    distance_matrix = np.load(output_dir / config["arch"] / f"distances_{dim}d.npy")

    with open(output_dir / 'train_indices.json', 'r') as fp:
        train_idx_full = json.load(fp)
    
    with open(output_dir / 'test_indices.json', 'r') as fp:
        test_idx_full = json.load(fp)
    
    with open(output_dir / config["arch"] / f'valid_indices_{dim}d.json', 'r') as fp:
        valid_idx = json.load(fp)
    
    train_idx = [i for i in train_idx_full if i in valid_idx]
    test_idx = [i for i in test_idx_full if i in valid_idx]

    train_distance_matrix = distance_matrix[train_idx,:]
    train_distance_matrix = train_distance_matrix[:, train_idx]
    np.fill_diagonal(train_distance_matrix, 1000)

    test_distance_matrix = distance_matrix[test_idx,:]
    test_distance_matrix = test_distance_matrix[:,train_idx]

    dataset = TUDataset(root=config['dataset_dir'], name=config['dataset_name'])

    train_labels = [dataset[i].y.item() for i in train_idx]
    test_labels = [dataset[i].y.item() for i in test_idx]

    scoring = 'f1' if config['average'] == 'binary' else 'f1_micro'

    ks = (3, 5, 7, 9, 11)
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
        logging.info(f"K={k}, Cross-validated F1 score: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    # retrain using best_k
    knn_test = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed')
    knn_test.fit(train_distance_matrix, train_labels)
    predictions = knn_test.predict(test_distance_matrix)
    f1 = f1_score(test_labels, predictions, average=config["average"])

    logging.info(f"The optimal number of K-nearest neighbors is {best_k} with a mean F1 score of {best_score:.4f}")
    logging.info(f"F1 Score on test set with K={best_k}: {f1:.4f}")


def svm_classifier(config, output_dir, dim):
    distance_matrix = np.load(output_dir / config["arch"] / f"distances_{dim}d.npy")

    with open(output_dir / 'train_indices.json', 'r') as fp:
        train_idx_full = json.load(fp)
    
    with open(output_dir / 'test_indices.json', 'r') as fp:
        test_idx_full = json.load(fp)
    
    with open(output_dir / config["arch"] / f'valid_indices_{dim}d.json', 'r') as fp:
        valid_idx = json.load(fp)
    
    train_idx = [i for i in train_idx_full if i in valid_idx]
    test_idx = [i for i in test_idx_full if i in valid_idx]

    train_distance_matrix = distance_matrix[train_idx,:]
    train_distance_matrix = train_distance_matrix[:, train_idx]
    np.fill_diagonal(train_distance_matrix, 1000)

    test_distance_matrix = distance_matrix[test_idx,:]
    test_distance_matrix = test_distance_matrix[:,train_idx]

    dataset = TUDataset(root=config['dataset_dir'], name=config['dataset_name'])

    train_labels = [dataset[i].y.item() for i in train_idx]
    test_labels = [dataset[i].y.item() for i in test_idx]

    scoring = 'f1' if config['average'] == 'binary' else 'f1_micro'
    
    gammas = [0.001, 0.01, 0.1, 1]
    cs = [0.1, 1, 10, 100]

    best_score = 0
    best_params = None
    best_kernel = None

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for gamma in gammas:
        # convert train distance matrix to (RBF) kernel matrix
        K_train = np.exp(-gamma * np.square(train_distance_matrix))

        for c in cs:
            svm = SVC(C=c, kernel='precomputed')
            scores = cross_val_score(svm,
                                     K_train,
                                     train_labels,
                                     cv=kf,
                                     scoring=scoring)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'C': c, 'gamma': gamma}
                best_kernel = K_train.copy()

    # convert test distance matrix to (RBF) kernel matrix
    K_test = np.exp(-best_params['gamma'] * np.square(test_distance_matrix))

    # retrain using best params
    svm_final = SVC(C=best_params['C'], kernel='precomputed')
    svm_final.fit(best_kernel, train_labels)
    predictions = svm_final.predict(K_test)
    f1 = f1_score(test_labels, predictions, average=config["average"])

    logging.info(f"The optimal C is {best_params['C']} and optimal gamma i {best_params['gamma']} with a mean F1 score of {best_score:.4f}")
    logging.info(f"F1 score on test set: {f1:.4f}")


def main(config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / config["arch"] / 'log_hull_classification.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Starting k-NN classification for dimension 2")
    knn_classifier(config, output_dir, 2)
    logging.info(f"Starting k-NN classification for dimension 3")
    knn_classifier(config, output_dir, 3)
    logging.info("k-NN classification completed.")

    logging.info(f"Starting SVM classification for method dimension 2")
    svm_classifier(config, output_dir, 2)
    logging.info(f"Starting SVM classification for method dimension 3")
    svm_classifier(config, output_dir, 3)
    logging.info("SVM classification completed.")
 

if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)