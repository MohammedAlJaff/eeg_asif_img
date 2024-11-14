import torch
import numpy as np
from tqdm import tqdm
import json
import os

import src.datasets.eeg_image as eimg


def save_config(loaded_config, root_path, filename='config_run.json'):
    with open(os.path.join(root_path, filename), 'w') as file:
        json.dump(vars(loaded_config), file)

def load_dataset(dataset_name, data_path, **kwargs):
    if dataset_name == "spampinato_npy":
        data_configs = {
            "t_l": 0.02,
            "t_h": 0.46,
            "fs": 1000,
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
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img']
        )
    elif dataset_name == "spampinato":
        data_configs = {
            "t_l": 0.02,
            "t_h": 0.46,
            "fs": 1000,
            "n_samples": 440,
            "n_channels": 128,
            "n_classes": 40,
        }
        dataset = eimg.SpampinatoDataset(
            data_path=data_path,
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img']
        )
    elif dataset_name == "things-eeg-2":
        select_channels = kwargs['select_channels'] if 'select_channels' in kwargs.keys() else None
        data_configs = {
            "t_l": -0.2,
            "t_h": 0.8,
            "fs": 128,  # I have changed this from 100 to 128 in the Dataset description
            "n_samples": 128,
            "n_channels": 17 if select_channels is None else len(select_channels),
            "n_classes": 1654,
        }
        test = kwargs['test'] if 'test' in kwargs.keys() else False
        print("TEST = ", test)
        dataset = eimg.ThingsEEG2(
            data_path=data_path,
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img'],
            pretrain_eeg=kwargs['pretrain_eeg'],
            test=test,
            select_channels=select_channels,
            training_ratio=kwargs['subj_training_ratio'],
            load_img_embedding=kwargs['load_img_embedding'],
            img_encoder=kwargs['img_encoder']
        )
    else: 
        raise NotImplementedError
    return dataset, data_configs

def get_embeddings(model, data_loader, modality="eeg", save=False, save_path=None, device='cuda'):
    
    # progress_bar = tqdm(data_loader)
    embeddings = None
    labels = None
    if model is not None:
        model.eval()
    with torch.no_grad():
        for i, (data, y) in enumerate(data_loader):
            if modality == "eeg":
                x, _ = data
            else:
                _, x = data
            x = x.to(device)
            y = y.to(device)
            if model is not None:
                e = model(x)
            else:
                e = x
            e = torch.nn.functional.normalize(e, p=2, dim=-1)
            if embeddings is None:
                embeddings = e.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()
            else:
                embeddings = np.concatenate((embeddings, e.detach().cpu().numpy()), axis=0)
                labels = np.concatenate((labels, y.detach().cpu().numpy()), axis=0)
            # embeddings.append(e.detach().cpu().numpy())
            # labels.append(y.detach().cpu().numpy())
    if save:
        print("Saving the Embeddings")
        if save_path:
            np.save(save_path, embeddings)
        else:
            np.save("./embeddings.py", embeddings)
    print(embeddings.shape)
    print(labels)
    return embeddings, labels

    