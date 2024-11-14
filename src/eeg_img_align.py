import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img")

import itertools
import torch
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import wandb
import argparse
import os
import warnings
import pickle

import src.utils as utils
from src.models.eeg_encoder import EEGEncoder
from src.models.eeg_classifier import EEGClassifier
from src.models.image_encoder import ImageEncoder
from src.models.trainer import BimodalTrainer, UnimodalTrainer
from src.datasets.eeg_image import Splitter
from src.models.training_utils import CLIPLoss, SoftCLIPLoss
from src import downstream


model_configs = {
        'eegnet': {},
        'lstm': {'lstm_size': 128, 'lstm_layers': 1},
        'EEGChannelNet': {},
        'resnet1d': {},
        'dreamsim_clip_vitb32': {'embed_dim': 512},
    }

def seed_everything(seed_val):
    np.random.seed(seed_val)
    # random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--data_path', type=str, help="Path to the EEG data")
    parser.add_argument('--save_path', type=str, help="Path to save the model")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the pretrained model")
    parser.add_argument('--split_path', type=str, default="/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img/data/spampinato_et_al/block_splits_by_image_all.pth")
    parser.add_argument('--dataset', type=str, default="things-eeg-2")
    parser.add_argument('--subject_id', type=int, nargs='+', default=[0], help="Subject ID(s). Provide one or more subject IDs.")
    parser.add_argument('--test_subject', type=int, default=None)
    parser.add_argument('--subj_training_ratio', type=float, default=1, help="a ratio between 0 and 1 determining how much of participants training samples to be used")
    parser.add_argument('--channels', type=int, nargs='+', default=None)
    parser.add_argument('--n_classes', type=int, default=1654)
    parser.add_argument('--eeg_enc', type=str, default="resnet1d", help="EEG Encoder")
    parser.add_argument('--img_enc', type=str, default="CLIP_IMG", help="Image Encoder")
    parser.add_argument('--loss', type=str, default="clip-loss", help="Loss function")
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=None)
    parser.add_argument('--net_seq_length', type=int, nargs='+', default=None)
    parser.add_argument('--modality', type=str, default="eeg-img")
    parser.add_argument('--epoch', type=int, default=1000, help="Number of epochs for pretraining")
    parser.add_argument('--finetune_epoch',  type=int, default=50, help="Number of epochs for finetuning (if the downstream task is classification)")
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--downstream', type=str, default=None)
    parser.add_argument('--separate_test', action="store_true")
    parser.add_argument('--precompute_img_emb', action="store_true")
    parser.add_argument('-b', '--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def return_dataloaders(dataset_nm, data_pth, sid, n_classes, batch, num_workers, seed_val, split_path, device_type, separate_test=False, **kwargs):

    data, ds_configs = utils.load_dataset(
        dataset_name=dataset_nm, data_path=paths['eeg_data'], n_classes=n_classes, sid=sid, load_img=kwargs['load_img'], 
        pretrain_eeg=kwargs['pretrain_eeg'], select_channels=kwargs['select_channels'], subj_training_ratio=kwargs['subj_training_ratio'],
        load_img_embedding=kwargs['load_img_embedding'], img_encoder=kwargs['img_enc_name'])
    print(ds_configs)
    
    g = torch.Generator().manual_seed(seed_val)

    if dataset_nm == "spampinato":
        train_dl = DataLoader(Splitter(data, split_path=split_path, split_num=0, split_name='train'), 
                                    batch_size=batch, drop_last=True, shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True if device_type == 'cuda:0' else False,
                                    generator=g)
        val_dl = DataLoader(Splitter(data, split_path=split_path, split_num=0, split_name='val'), 
                                    batch_size=batch, drop_last=False, shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True if device_type == 'cuda:0' else False,
                                    generator=g)
        test_dl = DataLoader(Splitter(data, split_path=split_path, split_num=0, split_name='test'), 
                                    batch_size=batch, drop_last=False, shuffle=False,
                                    pin_memory=True if device_type == 'cuda:0' else False,
                                    generator=g)
    else:
        if not separate_test:
            train_data, val_data, test_data = torch.utils.data.random_split(
                data, [0.8, 0.1, 0.1], generator=g)
        else:
            train_data, val_data = torch.utils.data.random_split(
                data, [0.8, 0.2], generator=g)
            test_data, _ = utils.load_dataset(dataset_name=dataset_nm, data_path=paths['eeg_data'], n_classes=n_classes, sid=kwargs['test_subject'], test=True, load_img=kwargs['load_img'], 
                                              pretrain_eeg=kwargs['pretrain_eeg'], select_channels=kwargs['select_channels'], subj_training_ratio=1.0,
                                              load_img_embedding=kwargs['load_img_embedding'], img_encoder=kwargs['img_enc_name'])
        train_dl = DataLoader(train_data, batch_size=batch, shuffle=True,
                                drop_last=True,
                                num_workers=num_workers,
                                pin_memory=True if device_type == 'cuda:0' else False,
                                generator=g)
        val_dl = DataLoader(val_data, batch_size=batch, shuffle=False,
                                drop_last=False,
                                num_workers=num_workers,
                                pin_memory=True if device_type == 'cuda:0' else False,
                                generator=g)
        test_dl = DataLoader(test_data, batch_size=batch, shuffle=False,
                                drop_last=False,
                                pin_memory=True if device_type == 'cuda:0' else False,
                                generator=g)
    return train_dl, val_dl, test_dl, ds_configs

if __name__ == "__main__":

    args = parse_args()
    with wandb.init():
        # args = wandb.config
        seed = args.seed
        dataset_name = args.dataset
        subject_id = args.subject_id
        if len(subject_id) == 1:    #TODO to be compatible with Spampinato until I fix it
            subject_id = subject_id[0]
        test_subject = args.test_subject
        n_classes = args.n_classes
        eeg_enc_name = args.eeg_enc
        img_enc_name = args.img_enc
        batch_size = args.batch
        lr = args.lr
        epochs = args.epoch
        finetune_epochs = args.finetune_epoch
        data_path = args.data_path
        save_path = args.save_path
        modality = args.modality
        downstream_task = args.downstream
        separate_test_set = args.separate_test
        channels=args.channels

        if args.subj_training_ratio == 0:
            epochs=0

        if args.net_filter_size:
            model_configs['resnet1d']['net_filter_size'] = args.net_filter_size

        if args.net_seq_length:
            model_configs['resnet1d']['net_seq_length'] = args.net_seq_length

        if separate_test_set and downstream_task == "classification":
            warnings.warn("The test set won't be used to finetune the classifier. seperate_test will be set to False")
            separate_test_set = False
        
        print("separate_test= ", separate_test_set)
        print("training subjects: ", subject_id)
        print("test subjects: ", test_subject if test_subject is not None else subject_id)

        # constants
        min_lr = 1e-07
        warmup_epochs = args.warmup
        weight_decay=0.1
        
        seed_everything(seed)
        paths = {"eeg_data": data_path, "save_path": save_path}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device = ", device)

        print("**********************************************************************************************")
        print(f"Starting a run on {dataset_name} with {eeg_enc_name}")
        print(f"Modalities Involved: {modality}")

        start_str = "scratch" if args.checkpoint is None else "pretrained"

        if modality == "eeg-img":
            if test_subject is None:
                directory_name = f"{start_str}_{dataset_name}_s{subject_id}_r{args.subj_training_ratio}_{eeg_enc_name}_{img_enc_name}_"
            else:
                directory_name = f"{start_str}_{dataset_name}_ts{test_subject}_r{args.subj_training_ratio}_{eeg_enc_name}_{img_enc_name}_"
        else:
            if test_subject is None:
                directory_name = f"{start_str}_{dataset_name}_s{subject_id}_r{args.subj_training_ratio}_{eeg_enc_name}_"
            else:
                directory_name = f"{start_str}_{dataset_name}_ts{test_subject}_r{args.subj_training_ratio}_{eeg_enc_name}_"
        
        current_datetime = datetime.now()
        directory_name += current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        paths["save_path"] = os.path.join(paths["save_path"], directory_name)
        os.makedirs(paths["save_path"], exist_ok=True)
        print(f"Directory '{directory_name}' created.")
        utils.save_config(args, root_path=paths['save_path'])
        print(vars(args))

        train_data_loader, val_data_loader, test_data_loader, data_configs = return_dataloaders(
            dataset_nm=dataset_name, 
            data_pth=paths['eeg_data'], sid=subject_id, 
            test_subject=test_subject if test_subject is not None else subject_id,
            n_classes=n_classes, batch=batch_size, 
            num_workers=args.n_workers,
            seed_val=seed, 
            split_path=args.split_path, 
            load_img=True if modality == "eeg-img" else False,
            pretrain_eeg=True if modality == "eeg-eeg" else False,
            separate_test=separate_test_set,
            select_channels=channels,
            subj_training_ratio=args.subj_training_ratio if args.subj_training_ratio > 0 else 0.01,
            load_img_embedding=args.precompute_img_emb,
            img_enc_name=img_enc_name,
            device_type=device)    
        
        if modality == "eeg-img" and not args.precompute_img_emb:
            img_encoder = ImageEncoder(
                backbone=img_enc_name,
                embed_dim=None,
                add_ln_layer=False,
            )
            img_encoder = img_encoder.float()

            embedding_size = img_encoder.embed_dim
        elif modality == "eeg-img" and args.precompute_img_emb:
            img_encoder = None
            embedding_size = model_configs[img_enc_name]['embed_dim']
        else:
            img_encoder = None
            embedding_size = args.embed_dim
        
        print("eeg embedding size: ", embedding_size)
        eeg_encoder = EEGEncoder(
            embed_dim=embedding_size,
            backbone=eeg_enc_name,
            n_channels=data_configs["n_channels"],
            n_samples=data_configs["n_samples"],
            n_classes=n_classes,
            model_path=None,
            device=device, 
            **model_configs[eeg_enc_name]
            )
        eeg_encoder = eeg_encoder.float()

        if args.loss == "clip-loss":
            loss = CLIPLoss(temperature=args.temperature)
        elif args.loss == "soft-clip":
            loss = SoftCLIPLoss(temperature=args.temperature)
        else:
            loss = CLIPLoss(temperature=args.temperature)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)['model_state_dict']
            eeg_encoder.load_state_dict(checkpoint)
            eeg_encoder.to(device)
        
        if epochs > 0:
            # if modality == "eeg-img":
            #     optim = torch.optim.AdamW(itertools.chain(eeg_encoder.parameters(), img_encoder.parameters()), lr=min_lr, weight_decay=weight_decay)
            # else:
            optim = torch.optim.AdamW(eeg_encoder.parameters(), lr=min_lr, weight_decay=weight_decay)
            trainer = BimodalTrainer(
                eeg_encoder=eeg_encoder,
                image_encoder=img_encoder,
                optimizer=optim, 
                loss=loss, 
                epochs=epochs, 
                warmup_epochs=warmup_epochs,
                lr=lr, min_lr=min_lr,  
                mixed_precision=True,
                num_classes=n_classes,
                save_path=paths["save_path"], 
                filename=f'{eeg_enc_name}_{dataset_name}', 
                precompute_img_emb=args.precompute_img_emb,
                device=device
                )
            best_eeg_encoder = trainer.train(train_data_loader, val_data_loader)
            eeg_encoder.load_state_dict(best_eeg_encoder['model_state_dict']) # TODO What if we also train the image encoder (embedding layer)
            test_loss = trainer.evaluate(eeg_encoder, img_encoder, test_data_loader)
            print(f"Test Loss: {test_loss}")

        print(f"Performing the Downstream Task for S{test_subject if test_subject is not None else subject_id} (tr={args.subj_training_ratio})")
        if downstream_task == "classification":
            train_data_loader, val_data_loader, test_data_loader, data_configs = return_dataloaders(
                dataset_nm=dataset_name, 
                data_pth=paths['eeg_data'], sid=subject_id, 
                test_subject=test_subject if test_subject is not None else subject_id,
                n_classes=n_classes, batch=batch_size, 
                num_workers=args.n_workers,
                seed_val=seed, 
                split_path=args.split_path, 
                load_img=False,
                pretrain_eeg=False,
                separate_test=separate_test_set,
                select_channels=channels,
                subj_training_ratio=args.subj_training_ratio,
                load_img_embedding=args.precompute_img_emb,
                img_enc_name=img_enc_name,
                device_type=device)
            loaders = {'train': train_data_loader, 'val': val_data_loader, 'test': test_data_loader} 
            test_loss, test_acc = downstream.classification(
                loaders=loaders,
                eeg_enc_name=eeg_enc_name, 
                dataset_name=dataset_name, n_channels=data_configs['n_channels'], n_samples=data_configs['n_samples'], n_classes=n_classes, 
                finetune_epochs=finetune_epochs, warmup_epochs=20, lr=lr, min_lr=min_lr, weight_decay=weight_decay,
                save_path=paths['save_path'],
                pretrained_encoder=eeg_encoder, model_configs=model_configs, device=device
            )
        elif downstream_task == "retrieval":
            _, _, test_data_loader, data_configs = return_dataloaders(
                dataset_nm=dataset_name, 
                data_pth=paths['eeg_data'], sid=subject_id, 
                test_subject=test_subject if test_subject is not None else subject_id,
                n_classes=n_classes, batch=batch_size, 
                num_workers=args.n_workers,
                seed_val=seed, 
                split_path=args.split_path, 
                load_img=True,
                pretrain_eeg=False,
                separate_test=separate_test_set,
                select_channels=channels,
                subj_training_ratio=args.subj_training_ratio if args.subj_training_ratio > 0 else 0.01,
                load_img_embedding=args.precompute_img_emb,
                img_enc_name=img_enc_name,
                device_type=device)
            top1_acc, top3_acc, top5_acc = downstream.retrieval(eeg_encoder, img_encoder, test_data_loader, device=device)
            topk_scores = {
                'top1': top1_acc,
                'top3': top3_acc,
                'top5': top5_acc
            }
            with open(os.path.join(paths["save_path"], "topk_performances.pkl"), 'wb') as f:
                pickle.dump(topk_scores, f)
        else:
            print("No Downstream Task Selected. We Are Done!")
            

        # classifier_model = EEGClassifier(
        #     backbone=eeg_enc_name,
        #     n_channels=data_configs["n_channels"], 
        #     n_samples=data_configs["n_samples"], 
        #     n_classes=n_classes,
        #     pretrained_encoder=eeg_encoder,
        #     device=device, 
        #     **model_configs[eeg_enc_name])
        # classifier_model = classifier_model.float()

        # loss = torch.nn.CrossEntropyLoss()
        # optim = torch.optim.AdamW(classifier_model.parameters(), lr=min_lr, weight_decay=weight_decay)
        # trainer = UnimodalTrainer(model=classifier_model, 
        #                           optimizer=optim, 
        #                           loss=loss, 
        #                           epochs=finetune_epochs, 
        #                           warmup_epochs=10,
        #                           lr=lr, min_lr=min_lr,  
        #                           mixed_precision=True,
        #                           num_classes=n_classes,
        #                           save_path=paths["save_path"], 
        #                           filename=f'cls_{eeg_enc_name}_{dataset_name}.pth', 
        #                           device=device)
        # best_classifier = trainer.train(train_data_loader, val_data_loader)
        # classifier_model.load_state_dict(best_classifier['model_state_dict'])
        # test_loss, test_acc = trainer.evaluate(classifier_model, test_data_loader)
        # print(f"Test Loss: {test_loss} | Test Acc.: {test_acc}")
        


        

        



