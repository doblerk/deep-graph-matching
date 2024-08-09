#  GNN-GED Framework

This repository contains the implementation of GNN-GED framework to perform graph matching learning with graph neural networks.

## Installation

#### Prerequisites
 - Python 3.10

#### Install
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install required packages
pip3 install -r requirements.txt

# Install the Python package
python3 -m pip install -e .
```

## How to use

#### Train a GNN model
```bash
# Split the data set
python3 gnn_ged/utils/split_dataset.py --dataset_dir data/TUDataset/ --dataset_name MUTAG --output_dir res/MUTAG/

# Train the model
python3 gnn_ged/training/train_model.py --dataset_dir data/TUDataset --dataset_name MUTAG --arch gin --indices_dir res/MUTAG --output_dir res/MUTAG/GIN/raw/
```

#### Compute GED
```bash
python3 main.py --dataset_dir data/TUDataset/ --dataset_name MUTAG --output_dir res/MUTAG/GIN/raw/
```