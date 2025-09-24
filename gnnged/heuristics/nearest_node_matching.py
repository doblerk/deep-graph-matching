import sys
import json
import logging
import numpy as np
from pathlib import Path
from time import time
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.datasets import TUDataset
from gnnged.heuristics.data_utils import load_embeddings
from gnnged.heuristics.nearest_node_utils import KNNGraphClassifier


def main_split(config):
    """
    Computes k-nearest node matching for direct graph classification
    """
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log_knn_matching.txt'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading dataset and node embeddings...")
    dataset = TUDataset(root=config['dataset_dir'], name=config['dataset_name'])

    with open(Path(config['output_dir']) / 'train_indices.json', 'r') as fp:
        train_idx = json.load(fp)

    with open(Path(config['output_dir']) / 'test_indices.json', 'r') as fp:
        test_idx = json.load(fp)

    node_embeddings = load_embeddings(output_dir / 'node_embeddings.h5', len(dataset))

    classifier = KNNGraphClassifier(dataset, train_idx, test_idx, node_embeddings)

    for k in [1, 5]:
        logging.info(f"Running {k}-NN...")
        t0 = time()
        preds = classifier.predict(k=k, weighted=False)
        duration = time() - t0
        acc = classifier.evaluate(preds, config["average"])
        logging.info(f"{k}-NN Accuracy (F1): {acc:.4f}, computation time: {duration} seconds")

    logging.info("Running weighted 5-NN...")
    t0 = time()
    preds = classifier.predict(k=5, weighted=True)
    duration = time() - t0
    acc = classifier.evaluate(preds, config["average"])
    logging.info(f"5-WNN Accuracy (F1): {acc:.4f}, computation time: {duration} seconds")


def main(config):
    """
    Computes k-nearest node matching for direct graph classification
    """
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log_knn_matching.txt'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading dataset and node embeddings...")
    dataset = TUDataset(root=config['dataset_dir'], name=config['dataset_name'])
    labels = dataset._data.y.numpy()
    num_classes = dataset.num_classes
    
    average = 'binary' if num_classes == 2 else 'micro'

    node_embeddings = load_embeddings(output_dir / 'node_embeddings.h5', len(dataset))
    
    all_test_acc = {'1': [], '5': [], 'w5': []}
    all_test_f1 = {'1': [], '5': [], 'w5': []}
    
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        classifier = KNNGraphClassifier(dataset, train_idx, test_idx, node_embeddings)
	    
        logging.info(f"[Run {split_id+1}]")

        for k in [1, 5]:
            t0 = time()
            preds = classifier.predict(k=k, weighted=False)
            duration = time() - t0
            acc, f1 = classifier.evaluate(preds, average)
            all_test_acc[str(k)].append(acc)
            all_test_f1[str(k)].append(f1)
            logging.info(f"{k}-NN Accuracy={acc:.4f}, F1={f1:.4f}, computation time: {duration} seconds")
        
        t0 = time()
        preds = classifier.predict(k=5, weighted=True)
        duration = time() - t0
        acc, f1 = classifier.evaluate(preds, average)
        all_test_acc['w5'].append(acc)
        all_test_f1['w5'].append(f1)
        logging.info(f"5-WNN Accuracy={acc:.4f}, F1={f1:.4f}, computation time: {duration} seconds")
	    
    logging.info("Final results over 5 random splits:")
    for key in all_test_acc.keys():
        acc_mean, acc_std = np.mean(all_test_acc[key]), np.std(all_test_acc[key])
        f1_mean, f1_std = np.mean(all_test_f1[key]), np.std(all_test_f1[key])
        logging.info(f"{key}-NN -> Accuracy: {acc_mean:.4f} +/- {acc_std:.4f}, F1: {f1_mean:.4f} +/- {f1_std:.4f}")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)