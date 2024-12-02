import os
import json
import argparse
import torch
import importlib
import torch
import optuna

import numpy as np

from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, Constant


def reset_weights(m):
    '''Resets the weights during each fold to avoid weight leakage'''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(train_loader, device, optimizer, model, criterion):
    '''Trains the model on train set'''
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, z = model(data.x, data.edge_index, data.batch)
        loss = criterion(z, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(test_loader, device, model, criterion):
    '''Evaluates the model and returns the accuracy and loss'''
    model.eval()
    total_correct = 0
    total_loss = 0
    total_samples = 0

    for data in test_loader:
        data = data.to(device)
        _, z = model(data.x, data.edge_index, data.batch)

        loss = criterion(z, data.y)
        total_loss += loss.item() * data.y.size(0)

        preds = z.argmax(dim=1)
        total_correct += (preds == data.y).sum().item()

        total_samples += data.y.size(0)

    acc = total_correct / total_samples
    loss = total_loss / total_samples

    return acc, loss


def objective(trial):

    # Setup the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=False)
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 2, 5)

    # Load the dataset from TUDataset
    transform = NormalizeFeatures() if args.use_attrs else None
    dataset = TUDataset(root=args.dataset_dir,
                        name=args.dataset_name,
                        use_node_attr=args.use_attrs,
                        transform=transform)
    
    # Check if the data set contains unlabelled nodes
    if 'x' not in dataset[0]:
        dataset.transform = Constant(value=1.0)

    with open(os.path.join(args.indices_dir, 'train_indices.json'), 'r') as fp:
        train_idx = json.load(fp)

    train_dataset = dataset[train_idx]
    train_labels = [dataset[i].y.item() for i in train_idx]

    # Perform k-Fold CV
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits)

    val_accuracies = []
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, train_labels)):

      train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )

      val_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )

      # Initialize the model
      model_module = importlib.import_module(f'gnn_ged.models.{args.arch}')
      model = model_module.Model(
              input_dim=dataset.num_features,
              hidden_dim=hidden_dim,
              n_classes=dataset.num_classes,
              n_layers=num_layers,
      ).to(device)

      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
      criterion = torch.nn.CrossEntropyLoss()

      for epoch in range(11):
        train(train_loader, device, optimizer, model, criterion)

      val_accuracy, val_loss = test(val_loader, device, model, criterion)
      val_accuracies.append(val_accuracy)
      val_losses.append(val_loss)

    return np.mean(val_accuracies)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--arch', type=str, choices=['gin', 'gat', 'gcn', 'gsage'], help='GNN architecture')
    parser.add_argument('--use_attrs', action='store_true', help='Use node attributes')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    # Create Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=11) # Number of trials

    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    with open(os.path.join(args.output_dir, 'log_cv.txt'), 'a') as file:
        file.write(f"lr: {trial.params['lr']}\n")
        file.write(f"weight_decay: {trial.params['weight_decay']}\n")
        file.write(f"batch_size: {trial.params['batch_size']}\n")
        file.write(f"hidden_dim: {trial.params['hidden_dim']}\n")
        file.write(f"num_layers: {trial.params['num_layers']}\n")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)