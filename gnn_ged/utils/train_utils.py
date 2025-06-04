import os
import h5py
import torch
from torch_geometric.loader import DataLoader
from typing import Tuple


def get_batch_size(dataset_len: int,
                   target_fraction: float = 0.4,
                   batch_size_ranges: Tuple[int] = (64, 128, 256, 512)):
    """
    Generalized batch size calculation.
    """
    estimated = dataset_len * target_fraction
    closest = min(batch_size_ranges, key=lambda x: abs(x - estimated))
    return closest


def get_best_trial_params(log_path):
    params = dict()
    with open(log_path, 'r') as file:
        for line in file:
            line = line.strip()
            key, value = line.split(':', 1)
            key, value = key.strip(), value.strip()
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            params[key] = value
    return params


@torch.no_grad()
def extract_embeddings(dataset, device, model, config):
    '''Extracts the node embeddings of the final layer and stores them in HDF'''
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embeddings = list()
    model.eval()
    for i, data in enumerate(data_loader, start=0):
        data = data.to(device)
        h, _ = model(data.x, data.edge_index, data.batch)
        embeddings.append(h.detach().cpu().numpy())
    
    with h5py.File(os.path.join(config['output_dir'], 'node_embeddings.h5'), 'w') as f:
        for i, mbddg in enumerate(embeddings, start=0):
            f.create_dataset(f'embedding_{i}', data=mbddg)