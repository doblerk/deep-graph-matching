import sys
import json
import logging
import numpy as np
from pathlib import Path
from time import time
from torch_geometric.datasets import TUDataset
from gnnged.heuristics.data_utils import load_embeddings
from gnnged.heuristics.nearest_node_utils import KNNGraphClassifier


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
        acc = classifier.evaluate(preds)
        logging.info(f"{k}-NN Accuracy (F1): {acc:.4f}, computation time: {duration} seconds")

    logging.info("Running weighted 5-NN...")
    t0 = time()
    preds = classifier.predict(k=5, weighted=True)
    duration = time() - t0
    acc = classifier.evaluate(preds)
    logging.info(f"5-WNN Accuracy (F1): {acc:.4f}, computation time: {duration} seconds")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)