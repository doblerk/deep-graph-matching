#  GNN-GED Framework
This repository contains an implementation of a GNN-based framework to derive graph edit operations.

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
 - Python >= 3.10
 - PyTorch & torch-geometric (see installation below)

#### Install
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch and torch-geometric (CUDA 12.4 shown, adjust if needed)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install torch_geometric

# Install the package in editable/development mode
python3 -m pip install -e .
```

## Usage

#### Run scripts with JSON parameter files
```json
{
    "dataset_dir": "data/TUDataset",
    "dataset_name": "MUTAG",
    "output_dir": "./res/MUTAG/",
    "use_attrs": false,
}
```

#### Split the data set
```bash
# Split the data set
python3 scripts/run_preprocessing.py
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
# Compute distances
python3 scripts/run_matching.py
```

#### Classify graphs
```bash
# Classify graphs
python3 scripts/run_evaluation.py
```

## Further information
Please refer to [Graph Matching](https://github.com/doblerk/graph-matching.git) for a faster alternative.