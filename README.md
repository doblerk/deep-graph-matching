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
```

## How to use

#### Train a GNN model
```bash
python3 gnn_ged/training/train_model.py --dataset_dir data/TUDataset --dataset_name MUTAG --arch gin --output_dir res/MUTAG/GIN/raw/
```

#### Compute GED
```bash
python3 main.py --dataset_dir data/TUDataset/ --dataset_name MUTAG --output_dir res/MUTAG/GIN/raw/
```