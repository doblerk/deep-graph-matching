#  GNN-GED Framework
This repository contains the implementation of a GNN-based framework to derive graph edit distance.

## Folder structure
```bash
├── data
│   └── TUDataset
│       └── MUTAG
│           └── raw
│
├── gnnged
│   ├── assignment
│   ├── edit_cost
│   ├── evaluation
|   ├── heuristics
│   ├── models
│   ├── training
│   └── utils
│
├── scripts
|
├── src
│   └── build
│ 
├── res
│   └── MUTAG
│       └── GIN
│           └── raw
│
├── tests
└── venv
```

## Installation

#### Prerequisites
 - Python 3.10

#### Install
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Check your CUDA configuration and install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install torch_geometric

# Install the Python package
python3 -m pip install -e .
```

## How to use

#### Split the data set
```bash
# Split the data set
python3 gnnged/utils/split_dataset.py --dataset_dir data/TUDataset/ --dataset_name MUTAG --output_dir res/MUTAG/
```

#### Train a GNN model
```bash
# Finetune the model
python3 scripts/run_finetuning.py

# Train the model
python3 scripts/run_training.py
```

#### Compute GED
```bash
python3 scripts/run_matching.py
```

#### Classify graphs
```bash
python3 gnn_ged/evaluation/graph_classification.py --distance_matrix res/MUTAG/GIN/raw/all_distances.npy --indices_dir res/MUTAG/ --dataset_dir data/TUDataset/ --dataset_name MUTAG
```

## Further information
Please refer to [Graph Matching](https://github.com/doblerk/graph-matching.git) for a faster alternative.