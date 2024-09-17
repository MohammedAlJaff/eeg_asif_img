import torch
import numpy as np
from tqdm import tqdm
import json
import os

import src.datasets.eeg_image as eimg


def save_config(loaded_config, root_path, filename='config_run.json'):
    with open(os.path.join(root_path, filename), 'w') as file:
        json.dump(loaded_config.as_dict(), file)

def load_dataset(dataset_name, data_path, **kwargs):
    if dataset_name == "spampinato_npy":
        data_configs = {
            "t_l": 0.02,
            "t_h": 0.46,
            "fs": 1000,
            # "n_samples": int((0.46 - 0.02) * 1000),
            "n_samples": 512,
            "n_channels": 128,
            "n_classes": 40,
        }
        dataset = eimg.EEGImagenet(
            data_path=data_path,
            n_classes=kwargs['n_classes'],
            time_low=data_configs['t_l'],
            time_high=data_configs['t_h'],
            fs=data_configs['fs'],
            subject_id=kwargs['sid']
        )
    else: 
        raise NotImplementedError
    return dataset, data_configs

def get_embeddings(model, data_loader, device='cuda'):
    
    progress_bar = tqdm(data_loader)
    embeddings = None
    labels = None
    model.eval()
    with torch.no_grad():
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)
            e = model(x)
            if embeddings is None:
                embeddings = e.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()
            else:
                embeddings = np.concatenate((embeddings, e.detach().cpu().numpy()), axis=0)
                labels = np.concatenate((labels, y.detach().cpu().numpy()), axis=0)
            # embeddings.append(e.detach().cpu().numpy())
            # labels.append(y.detach().cpu().numpy())
    print(embeddings.shape)
    print(labels)
    return embeddings, labels

    