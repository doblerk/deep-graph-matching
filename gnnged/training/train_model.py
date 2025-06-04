import os
import json
import argparse
import datetime
import importlib

import torch

from time import time
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, Constant

from gnnged.utils.train_utils import get_batch_size, \
                                      get_best_trial_params, \
                                      extract_embeddings


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


def train_model(train_loader, test_loader, device, optimizer, model, criterion, scheduler, config):
    '''Trains the model, reports the accuracy and loss, and saves the last ckpt'''
    t0 = time()

    for epoch in range(11):

        train(train_loader, device, optimizer, model, criterion)

        scheduler.step()

        if epoch % 10 == 0:
            train_accuracy, train_loss = test(train_loader, device, model, criterion)
            test_accuracy, test_loss = test(test_loader, device, model, criterion)
            print(f'Epoch {epoch:<3} | Train Loss: {train_loss:.5f} | Train Acc: {train_accuracy*100:.2f} | Test Loss: {test_loss:.5f} | Test Acc: {test_accuracy*100:.2f}')
        
        log_stats = {'Epoch': epoch, 'Train loss': train_loss, 'Train accuracy': train_accuracy, 'Test loss': test_loss, 'Test accuracy': test_accuracy}
        with open(os.path.join(config['output_dir'], 'log.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    
    t1 = time()
    computation_time = str(datetime.timedelta(seconds=int(t1 - t0)))
    print(f'Training time {computation_time}')

    # Save the model
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
    }
    torch.save(save_dict, os.path.join(config['output_dir'], 'checkpoint.pth'))


# def get_args_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
#     parser.add_argument('--dataset_name', type=str, help='Dataset name')
#     parser.add_argument('--arch', type=str, choices=['gin', 'gat', 'gcn', 'gsage'], help='GNN architecture')
#     parser.add_argument('--use_attrs', type=bool, default=False, help='Use node attributes')
#     parser.add_argument('--indices_dir', type=str, help='Path to indices')
#     parser.add_argument('--output_dir', type=str, help='Path to output directory')
#     return parser


def main(config):

    params = get_best_trial_params(os.path.join(config['output_dir'], 'log_cv.txt'))

    # Write logs
    # log_args = {k:str(v) for (k,v) in sorted(dict(vars(args)).items())}
    with open(os.path.join(config['output_dir'], 'log.txt'), 'a') as f:
        # f.write(json.dumps(log_args) + '\n')
        f.write(json.dumps(params) + '\n')
    
    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset from TUDataset
    transform = NormalizeFeatures() if config['use_attrs'] else None
    dataset = TUDataset(root=config['dataset_dir'],
                        name=config['dataset_name'],
                        use_node_attr=config['use_attrs'],
                        transform=transform)
    
    # Check if the data set contains unlabelled nodes
    if 'x' not in dataset[0]:
        dataset.transform = Constant(value=1.0)
    
    # Load the train and test indices
    with open(os.path.join(config['indices_dir'], 'train_indices.json'), 'r') as fp:
        train_idx = json.load(fp)
    
    with open(os.path.join(config['indices_dir'], 'test_indices.json'), 'r') as fp:
        test_idx = json.load(fp)

    # Prepare the data
    train_dataset, test_dataset = dataset[train_idx], dataset[test_idx]

    batch_size = get_batch_size(len(train_dataset))
    
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
    )
    
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
    )
    
    # Initialize the model
    model_module = importlib.import_module(f"gnn_ged.models.{config['arch']}")
    model = model_module.Model(
            input_dim=dataset.num_features,
            hidden_dim=params['hidden_dim'],
            n_classes=dataset.num_classes,
            n_layers=params['num_layers'],
    ).to(device)

    # Define the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])

    # Train the model
    train_model(train_loader, test_loader, device, optimizer, model, criterion, scheduler, config)
    
    # Extract the embeddings
    extract_embeddings(dataset, device, model, config)


if __name__ == '__main__':
    with open('params.json', 'r') as f:
        config = json.load(f)
    main(config)