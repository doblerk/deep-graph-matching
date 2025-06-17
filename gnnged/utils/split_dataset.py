import os
import sys
import json
from pathlib import Path
import numpy as np
from torch_geometric.datasets import TUDataset


def main(config):

    # Load the dataset from TUDataset
    dataset = TUDataset(root=config["dataset_dir"], name=config["dataset_name"])
    
    # Shuffle the indices
    dataset_idx = list(range(len(dataset)))
    np.random.shuffle(dataset_idx)

    # Split the dataset as 80-20 train and test
    train_idx = sorted(dataset_idx[:int(len(dataset)*0.8)])
    test_idx = sorted(dataset_idx[int(len(dataset)*0.8):])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the indices
    with open(output_dir / 'train_indices.json', 'w') as fp:
        json.dump(train_idx, fp)
    
    with open(output_dir / 'test_indices.json', 'w') as fp:
        json.dump(test_idx, fp)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'params.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config file '{config_path}': {e}")
        sys.exit(1)

    main(config)