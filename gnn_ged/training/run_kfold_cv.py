import os
import json
import argparse
import torch
import importlib

import numpy as np

from torch.optim.lr_scheduler import StepLR
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


def perform_kfold_cv(dataset, train_dataset, train_labels, device, args):
    '''Performs k-fold cross validation'''

    # Perform K-fold cross validation
    k_fold = 10
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=None)

    # Perform k-fold cross validation to get an estimate of the model's accuracy
    val_accuracies = []
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, train_labels)):
        print(f'--------\nFold {fold + 1}\n--------')

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        
        val_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )

        # Initialize the model
        model_module = importlib.import_module(f'gnn_ged.models.{args.arch}')
        model = model_module.Model(
                input_dim=dataset.num_features,
                hidden_dim=args.hidden_dim,
                n_classes=dataset.num_classes,
                n_layers=args.n_layers,
        ).to(device)

        # Define the optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        # Reset the weights
        # model.apply(reset_weights)

        # Train the model on the train set
        for epoch in range(args.epochs):
            train(train_loader, device, optimizer, model, criterion)
            scheduler.step()

        # Evaluate the model on the test set
        val_accuracy, val_loss = test(val_loader, device, model, criterion)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        # Write logs
        log_stats = {'Fold': fold+1, 'Validation accuracy': val_accuracy}
        with open(os.path.join(args.output_dir, 'log_cv.txt'), 'a') as f:
           f.write(json.dumps(log_stats) + '\n')

    with open(os.path.join(args.output_dir, 'log_cv.txt'), 'a') as f:
        f.write(json.dumps({'Validation accuracies': val_accuracies}) + '\n')
        f.write(json.dumps({'Mean accuracy': np.mean(val_accuracies)}) + '\n')
        f.write(json.dump({'Std accuracy': np.std(val_accuracies)}) + '\n')


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--arch', type=str, choices=['gin', 'gat', 'gcn', 'gsage'], help='GNN architecture')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden channel dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of convolution layers')
    parser.add_argument('--epochs', type=int, default=201, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use_attrs', action='store_true', help='Use node attributes')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):

    log_args = {k:str(v) for (k,v) in sorted(dict(vars(args)).items())}
    with open(os.path.join(args.output_dir, 'log_cv.txt'), 'a') as f:
        f.write(json.dumps(log_args) + '\n')

    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_labels = [dataset[i].y.item() for i in train_idx]

    # Perform k-fold cross-testidation
    train_dataset = dataset[train_idx]
    perform_kfold_cv(dataset, train_dataset, train_labels, device, args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)