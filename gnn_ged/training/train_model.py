import os
import json
import argparse
import datetime
import importlib

import torch
import pickle
import numpy as np

from time import time
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures


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
    acc = 0
    loss = 0
    for data in test_loader:
        data = data.to(device)
        _, z = model(data.x, data.edge_index, data.batch)
        acc += int((z.argmax(dim=1) == data.y).sum()) / len(test_loader.dataset)
        loss += criterion(z, data.y) * len(data) / len(test_loader.dataset)
    return acc, loss


@torch.no_grad()
def extract_embeddings(train_dataset, train_idx, test_dataset, test_idx, device, model, args):
    '''Extracts the node embeddings of the final layer and returns a dictionnary with key-value pair as graph idx: embedding'''
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_dic = {}
    model.eval()
    for i, data in enumerate(train_loader, start=0):
        data = data.to(device)
        h, _ = model(data.x, data.edge_index, data.batch)
        train_dic[train_idx[i]] = h.detach().cpu().numpy()
        np.save(os.path.join(args.output_dir, f'embeddings/embedding_{train_idx[i]}.npy'), h.detach().cpu().numpy())
    
    with open(os.path.join(args.output_dir, 'train_embeddings.pkl'), 'wb') as fp:
        pickle.dump(train_dic, fp)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_dic = {}
    model.eval()
    for i, data in enumerate(test_loader, start=0):
        data = data.to(device)
        h, _ = model(data.x, data.edge_index, data.batch)
        test_dic[test_idx[i]] = h.detach().cpu().numpy()
        np.save(os.path.join(args.output_dir, f'embeddings/embedding_{test_idx[i]}.npy'), h.detach().cpu().numpy())

    with open(os.path.join(args.output_dir, 'test_embeddings.pkl'), 'wb') as fp:
        pickle.dump(test_dic, fp)


def train_model(train_loader, test_loader, device, optimizer, model, criterion, scheduler, args):
    '''Trains the model, reports the accuracy and loss, and saves the last ckpt'''
    t0 = time()
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(args.epochs):

        train(train_loader, device, optimizer, model, criterion)

        train_accuracy, train_loss = test(train_loader, device, model, criterion)
        test_accuracy, test_loss = test(test_loader, device, model, criterion)

        scheduler.step()

        if epoch % 1 == 0:
            print(f'Epoch {epoch:<3} | Train Loss: {train_loss:.5f} | Train Acc: {train_accuracy*100:.2f} | Test Loss: {test_loss:.5f} | Test Acc: {test_accuracy*100:.2f}')
        
        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        
        log_stats = {'Epoch': epoch, 'Train loss': train_loss.item(), 'Train accuracy': train_accuracy, 'Test loss': test_loss.item(), 'Test accuracy': test_accuracy}
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
        
        # if patience_counter >= args.patience:
        #     print(f'Early stopping at epoch {epoch}')
        #     break
    
    t1 = time()
    computation_time = str(datetime.timedelta(seconds=int(t1 - t0)))
    print(f'Training time {computation_time}')

    # Save the model
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
    }
    torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))


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
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--use_attrs', action='store_true', help='Use node attributes')
    parser.add_argument('--indices_dir', type=str, help='Path to indices')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):

    # Write logs
    log_args = {k:str(v) for (k,v) in sorted(dict(vars(args)).items())}
    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
        f.write(json.dumps(log_args) + '\n')
    
    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset from TUDataset
    if args.use_attrs:
        dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name, use_node_attr=True, transform=NormalizeFeatures())
    else:
        dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)
    
    train_idx = np.load(os.path.join(args.indices_dir, 'train_indices.npy'))
    test_idx = np.load(os.path.join(args.indices_dir, 'test_indices.npy'))

    train_dataset, test_dataset = dataset[train_idx], dataset[test_idx]
    
    # Prepare the data
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
    )
    
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
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

    # Train the model
    train_model(train_loader, test_loader, device, optimizer, model, criterion, scheduler, args)
    
    # Extract the embeddings
    extract_embeddings(train_dataset, train_idx, test_dataset, test_idx, device, model, args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)