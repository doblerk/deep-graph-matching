import sys
import json
import logging
import torch
import importlib
import torch
import optuna

import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, Constant

from gnnged.utils.train_utils import get_batch_size


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
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=32)
    num_layers = trial.suggest_int('num_layers', 2, 5, step=1)
    step_size = trial.suggest_int('step_size', 10, 50, step=10)
    gamma = trial.suggest_float('gamma', 5e-1, 9e-1, step=1e-1)

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

    with open(Path(config['output_dir']) / 'train_indices.json', 'r') as fp:
        train_idx = json.load(fp)

    train_labels = [dataset[i].y.item() for i in train_idx]

    # Perform k-Fold CV
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    val_accuracies = []

    for fold, (sub_train_idx, val_idx) in enumerate(kf.split(train_idx, train_labels)):
        
        train_indices = [train_idx[i] for i in sub_train_idx]
        val_indices = [train_idx[i] for i in val_idx]

        batch_size = get_batch_size(len(train_indices))

        train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_indices),
            )
        
        val_loader = DataLoader(
                dataset=dataset,
                batch_size=len(val_idx),
                sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            )
        
        # Initialize the model
        model_module = importlib.import_module(f"gnnged.models.{config['arch']}")
        model = model_module.Model(
                input_dim=dataset.num_features,
                hidden_dim=hidden_dim,
                n_classes=dataset.num_classes,
                n_layers=num_layers,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        for epoch in range(1):
            train(train_loader, device, optimizer, model, criterion)
            scheduler.step()

            if fold == 0:
                # first fold: report and prune
                _, val_loss = test(val_loader, device, model, criterion)
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        val_accuracy, _ = test(val_loader, device, model, criterion)
        val_accuracies.append(val_accuracy)

    trial.set_user_attr('val_accuracies', val_accuracies)

    return np.mean(val_accuracies)


def main(config):
    output_dir = Path(config['output_dir']) / config['arch']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "log_finetuning.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting Optuna study...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1, timeout=82800) # Number of trials

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    logging.info("Study statistics:")
    logging.info(f"  Number of finished trials: {len(study.trials)}")
    logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"  Number of complete trials: {len(complete_trials)}")

    best_trial = study.best_trial
    logging.info("Best trial:")
    logging.info(f"  Value: {best_trial.value}")
    logging.info("  Params:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    best_val_accuracies = best_trial.user_attrs['val_accuracies']
    logging.info(f"Validation accuracies: {best_val_accuracies}")

    best_val_accuracies = best_trial.user_attrs['val_accuracies']

    mean_accuracy = np.mean(best_val_accuracies)
    std_accuracy = np.std(best_val_accuracies)

    logging.info(f"Mean k-fold CV accuracy: {mean_accuracy:.4f}")
    logging.info(f"Standard deviation of k-fold CV accuracy: {std_accuracy:.4f}")
    
    best_params_file = output_dir / 'best_params.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_trial.params, f, indent=2)
        

if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)