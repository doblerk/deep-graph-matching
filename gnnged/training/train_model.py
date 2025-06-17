import sys
import json
import logging
import datetime
import importlib
import torch

from pathlib import Path
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


def train_model(train_loader, test_loader, device, optimizer, model, criterion, scheduler, config, output_dir):
    '''Trains the model, reports accuracy and loss, saves checkpoint, and logs stats'''

    t0 = time()

    for epoch in range(1):
        train(train_loader, device, optimizer, model, criterion)
        scheduler.step()

        if epoch % 10 == 0:
            train_accuracy, train_loss = test(train_loader, device, model, criterion)
            test_accuracy, test_loss = test(test_loader, device, model, criterion)
            
            log_stats = {
                'Epoch': epoch,
                'Train loss': train_loss,
                'Train accuracy': train_accuracy,
                'Test loss': test_loss,
                'Test accuracy': test_accuracy
            }
            logging.info(json.dumps(log_stats))
    
    t1 = time()
    computation_time = str(datetime.timedelta(seconds=int(t1 - t0)))
    logging.info(f'Training completed in {computation_time}')

    # Save the model checkpoint
    checkpoint_path = output_dir / 'checkpoint.pth'
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
    }
    torch.save(save_dict, checkpoint_path)
    logging.info(f'Model checkpoint saved at {checkpoint_path}')


def main(config):
    # Setup logging
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'log_training.txt'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    best_params_file = output_dir / 'best_params.json'
    with open(best_params_file, 'r') as f:
        params = json.load(f)

    # Log params as JSON string
    logging.info("Best trial params: %s", json.dumps(params))
    
    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load the dataset from TUDataset
    transform = NormalizeFeatures() if config['use_attrs'] else None
    dataset = TUDataset(root=config['dataset_dir'],
                        name=config['dataset_name'],
                        use_node_attr=config.get('use_attrs', False),
                        transform=transform)
    logging.info(f"Dataset loaded: {config['dataset_name']} with {len(dataset)} samples.")
    
    # Check for unlabelled nodes
    if not hasattr(dataset[0], 'x') or dataset[0].x is None:
        dataset.transform = Constant(value=1.0)
        logging.info("Dataset missing node features 'x', applied Constant transform.")
    
    # Load the train and test indices
    with open(Path(config['output_dir']) / 'train_indices.json', 'r') as fp:
        train_idx = json.load(fp)
    with open(Path(config['output_dir']) / 'test_indices.json', 'r') as fp:
        test_idx = json.load(fp)
    logging.info(f'Loaded train/test indices: {len(train_idx)}/{len(test_idx)}')

    # Prepare datasets and loaders
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

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
    logging.info(f'Train loader batch size: {batch_size}, Test loader batch size: {len(test_dataset)}')

    # Initialize model dynamically
    model_module = importlib.import_module(f"gnnged.models.{config['arch']}")
    model = model_module.Model(
            input_dim=dataset.num_features,
            hidden_dim=params['hidden_dim'],
            n_classes=dataset.num_classes,
            n_layers=params['num_layers'],
    ).to(device)
    logging.info(f'Model initialized: {config["arch"]}')

    # Setup optimizer, criterion, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    logging.info('Optimizer, criterion, and scheduler initialized.')

    # Train the model
    train_model(train_loader, test_loader, device, optimizer, model, criterion, scheduler, config, output_dir)
    logging.info('Training completed.')
    
    # Extract the embeddings
    extract_embeddings(dataset, device, model, config)
    logging.info('Embeddings extraction completed.')


if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)