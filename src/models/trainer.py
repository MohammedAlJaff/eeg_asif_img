import os
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
import src.models.training_utils as ut
import copy 


class UnimodalTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss,
                 save_path, filename, patience=25, device='cuda:0', **kwargs):

        self.device = device

        self.model = model.to(device)
        self.loss = loss.to("cuda" if device.startswith("cuda") else "cpu")
        self.optimizer = optimizer
        self.accuracy = Accuracy(task='multiclass', num_classes=kwargs["num_classes"])

        self.epochs = kwargs['epochs']
        self.mixed_precision = kwargs['mixed_precision']
        self.lr = kwargs['lr']
        self.min_lr = kwargs['min_lr']
        self.warmup_epochs = kwargs['warmup_epochs']

        self.save_path = save_path
        self.filename = filename
        self.patience = patience
        self.clip_grad = 0.8
        self.log_every = 100

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        scaler = GradScaler(enabled=self.mixed_precision)

        warmup_scheduler = ut.WarmupScheduler(self.optimizer, self.warmup_epochs, self.lr, start_lr=self.min_lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.patience)

        best_model = None
        best_loss = 10000000
        patience = 150
        print("Training Started...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}/{self.epochs}.")
            steps = 0
            loss_epoch = []
            y_true = []
            y_pred = []
            self.model.train()
            progress_bar = tqdm(train_data_loader)
            for data, y in progress_bar:

                x = data

                self.optimizer.zero_grad()

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.mixed_precision):
                    preds = self.model(x)
                    loss = self.loss(preds, y)

                loss_epoch.append(loss.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))

                # self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(), clip_grad=self.clip_grad)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                with torch.no_grad():
                    train_acc = self.accuracy(torch.stack(y_pred).detach().cpu().float(),
                                              torch.stack(y_true).detach().cpu())

                steps += 1

            train_loss = np.mean(loss_epoch)

            val_loss, val_acc = self.evaluate(self.model, val_data_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = self.patience
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }
            # else:
            #     patience -= 1

            # if patience == 0:
            #     break

            # Warmup phase
            if epoch <= self.warmup_epochs:
                warmup_scheduler.step()  # Adjust learning rate during warmup
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # After warmup, apply ReduceLROnPlateau scheduler
            if epoch > self.warmup_epochs:
                lr_scheduler.step(val_loss)  # Reduce LR if val_loss does not improve
                current_lr = self.optimizer.param_groups[0]['lr']

            print(f'Epoch: {epoch}')
            print(f'Training Acc.: {train_acc}| Training Loss: {train_loss}')
            print(f'Validation Acc.: {val_acc}| Validation Loss: {val_loss}')

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": current_lr
            })

            if epoch % self.log_every == 0:
                torch.save(best_model, os.path.join(self.save_path, self.filename + f"_{epoch}" + ".pth"))


        print("Finished training.")
        print("Creating checkpoint.")

        if best_model is None:
            best_model = {
                'epoch': self.epochs,
                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }

        print(f"Best Validation Loss = {best_model['val_loss']} (Epoch = {best_model['epoch']})")
        # filename = f'{self.save_path}/checkpoint.pth.tar'
        torch.save(best_model, os.path.join(self.save_path, self.filename + f"_{epoch}" + ".pth"))
        # filename_img = f'{self.writer.log_dir}/checkpoint_image_encoder.pth.tar'
        # filename_eeg = f'{self.writer.log_dir}/checkpoint_eeg_encoder.pth.tar'
        # torch.save(self.image_encoder.state_dict(), filename_img)
        # torch.save(self.eeg_encoder.state_dict(), filename_eeg)
        print("Finished creating checkpoint.")

        return best_model

    def evaluate(self, model, dataloader):

        model.eval()
        with torch.no_grad():

            loss_epoch = []
            y_true = []
            y_pred = []
            progress_bar = tqdm(dataloader)
            for data, y in progress_bar:
                x = data
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.mixed_precision):
                    preds = self.model(x)
                    loss = self.loss(preds, y)
                loss_epoch.append(loss.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))
            mean_loss_epoch = np.mean(loss_epoch)
            acc = self.accuracy(torch.stack(y_pred).detach().cpu().float(),
                                torch.stack(y_true).detach().cpu())

        return mean_loss_epoch, acc 


class BimodalTrainer:
    def __init__(self, eeg_encoder: torch.nn.Module, image_encoder: torch.nn.Module, optimizer: torch.optim.Optimizer, loss,
                 save_path, filename, device='cuda:0', patience=25, return_subject_id=False, **kwargs):

        self.device = device

        self.eeg_encoder = eeg_encoder.to(device)
        self.image_encoder = image_encoder.to(device) if image_encoder is not None else None
        self.return_subject_id = return_subject_id

        self.loss = loss.to("cuda" if device.startswith("cuda") else "cpu")
        self.optimizer = optimizer

        self.epochs = kwargs['epochs']
        self.mixed_precision = kwargs['mixed_precision']
        self.lr = kwargs['lr']
        self.min_lr = kwargs['min_lr']
        self.warmup_epochs = kwargs['warmup_epochs']
        self.scheduler = kwargs['scheduler']
        self.initial_epochs = kwargs['initial_epochs'] if 'initial_epochs' in kwargs else 0
        self.common_params = kwargs['common_params']
        
        self.save_path = save_path
        self.filename = filename
        self.patience = patience
        self.clip_grad = 0.8
        self.log_every = 0

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        scaler = GradScaler(enabled=self.mixed_precision)

        warmup_scheduler = ut.WarmupScheduler(self.optimizer, self.warmup_epochs, self.lr, start_lr=self.min_lr)
        if self.scheduler == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.patience)
        elif self.scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=self.min_lr)
        else:
            raise NotImplementedError


        best_model = None
        best_loss = 10000000
        patience = 150
        print("Training Started...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}/{self.epochs}.")

            # Update subject-specific parameters only for initial epochs
            if self.common_params is not None:
                if epoch < self.initial_epochs:
                    for p in self.common_params:
                        p.requires_grad = False  # Freeze non-subject-specific weights
                else:
                    for p in self.common_params:
                        p.requires_grad = True   # Unfreeze all weights after initial epochs

            steps = 0
            loss_epoch = []
            self.eeg_encoder.train()
            if self.image_encoder is not None:
                # self.image_encoder.train()
                self.image_encoder.eval()
            progress_bar = tqdm(train_data_loader)
            for data, _ in progress_bar:

                self.optimizer.zero_grad()

                if self.return_subject_id:
                    eeg, image = data[0]
                    subject_id = data[1]
                else:
                    eeg, image = data
                eeg, image = eeg.to(self.device, non_blocking=True), image.to(self.device, non_blocking=True)

                # eeg = F.normalize(eeg, p=2, dim=-1)
                # image = F.normalize(image, p=2, dim=-1)

                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.mixed_precision):
                    if self.return_subject_id:
                        z_i = self.eeg_encoder(eeg, subject_id)
                    else:
                        z_i = self.eeg_encoder(eeg)
                    if self.image_encoder is not None:
                        z_j = self.image_encoder(image)
                    else:
                        z_j = self.eeg_encoder(image)
                    z_i = F.normalize(z_i, p=2, dim=-1)
                    z_j = F.normalize(z_j, p=2, dim=-1)
                    loss = self.loss(z_i, z_j)

                loss_epoch.append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if self.device == torch.device('cuda:0'):
                    self.lr = self.optimizer.param_groups[0]["lr"]

                steps += 1

            train_loss = np.mean(loss_epoch)

            val_loss = self.evaluate(self.eeg_encoder, self.image_encoder, val_data_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = self.patience
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(self.eeg_encoder.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'val_loss': val_loss,
                }
            # else:
            #     patience -= 1

            # if patience == 0:
            #     break

            # Warmup phase
            if epoch < self.warmup_epochs:
                warmup_scheduler.step()  # Adjust learning rate during warmup
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # After warmup, apply ReduceLROnPlateau scheduler
            if epoch >= self.warmup_epochs:
                lr_scheduler.step(val_loss)  # Reduce LR if val_loss does not improve
                current_lr = self.optimizer.param_groups[0]['lr']

            print(f'Epoch: {epoch}')
            print(f'Training Loss: {train_loss}| Validation Loss: {val_loss}')

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr
            })

            if self.log_every != 0 and epoch % self.log_every == 0:
                torch.save(best_model, os.path.join(self.save_path, self.filename + f"_{epoch}" + ".pth"))

        print("Finished training.")
        print("Creating checkpoint.")

        if best_model is None:
            best_model = {
                'epoch': self.epochs,
                'model_state_dict': copy.deepcopy(self.eeg_encoder.state_dict()),
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                'val_loss': val_loss,
            }

        print(f"Best Validation Loss = {best_model['val_loss']} (Epoch = {best_model['epoch']})")
        # filename = f'{self.save_path}/checkpoint.pth.tar'
        torch.save(best_model, os.path.join(self.save_path, self.filename + f"_{epoch}" + ".pth"))
        # filename_img = f'{self.writer.log_dir}/checkpoint_image_encoder.pth.tar'
        # filename_eeg = f'{self.writer.log_dir}/checkpoint_eeg_encoder.pth.tar'
        # torch.save(self.image_encoder.state_dict(), filename_img)
        # torch.save(self.eeg_encoder.state_dict(), filename_eeg)
        print("Finished creating checkpoint.")

        return best_model

    def evaluate(self, eeg_encoder, image_encoder, dataloader):

        eeg_encoder.eval()
        if image_encoder is not None:
            image_encoder.eval()
        with torch.no_grad():

            loss_epoch = []
            progress_bar = tqdm(dataloader)
            for data, y in progress_bar:
                if self.return_subject_id:
                    data, subject_id = data
                eeg, image = data
                eeg, image = eeg.to(self.device, non_blocking=True), image.to(self.device, non_blocking=True)
                
                # eeg = F.normalize(eeg, p=2, dim=-1)
                # image = F.normalize(image, p=2, dim=-1)

                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.mixed_precision):
                    if self.return_subject_id:
                        z_i = eeg_encoder(eeg, subject_id)
                    else:
                        z_i = eeg_encoder(eeg)
                    if image_encoder is not None:
                        z_j = image_encoder(image)
                    else:
                        z_j = eeg_encoder(image)
                    z_i = F.normalize(z_i, p=2, dim=-1)
                    z_j = F.normalize(z_j, p=2, dim=-1)

                    loss = self.loss(z_i, z_j)
                loss_epoch.append(loss.item())
                
            mean_loss_epoch = np.mean(loss_epoch)
            
        return mean_loss_epoch
