import os
import numpy as np
import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
import src.models.training_utils as ut
import copy 


class UnimodalTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss,
                 save_path, filename, device='cuda', **kwargs):

        self.device = device

        self.model = model.to(device)
        self.loss = loss.to(device)
        self.loss_scaler = ut.NativeScalerWithGradNormCount()
        self.optimizer = optimizer
        self.accuracy = Accuracy(task='multiclass', num_classes=kwargs["num_classes"])

        self.epochs = kwargs['epochs']
        self.mixed_precision = kwargs['mixed_precision']
        self.lr = kwargs['lr']
        self.min_lr = kwargs['min_lr']
        self.warmup_epochs = kwargs['warmup_epochs']

        self.save_path = save_path
        self.filename = filename
        self.patience = 100
        self.clip_grad = 0.8

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        scaler = GradScaler(self.device, enabled=self.mixed_precision)

        best_model = None
        best_loss = 10000000
        patience = self.patience
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

                # adjust learning rate
                ut.adjust_learning_rate(self.optimizer, steps / len(train_data_loader) + epoch,
                                        warmup_epochs=self.warmup_epochs, num_epoch=self.epochs,
                                        lr=self.lr, min_lr=self.min_lr)

                x, _ = data

                self.optimizer.zero_grad()

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                    preds = self.model(x)
                    loss = self.loss(preds, y)

                loss_epoch.append(loss.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))

                self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(), clip_grad=self.clip_grad)
                with torch.no_grad():
                    train_acc = self.accuracy(torch.stack(y_pred).detach().cpu().float(),
                                              torch.stack(y_true).detach().cpu())

                if self.device == torch.device('cuda'):
                    self.lr = self.optimizer.param_groups[0]["lr"]

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
            else:
                patience -= 1

            if patience == 0:
                break

            print(f'Epoch: {epoch}')
            print(f'Training Acc.: {train_acc}| Training Loss: {train_loss}')
            print(f'Validation Acc.: {val_acc}| Validation Loss: {val_loss}')

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            })

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
        torch.save(best_model, os.path.join(self.save_path, self.filename))
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
                x, _ = data
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                    preds = self.model(x)
                    loss = self.loss(preds, y)
                loss_epoch.append(loss.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))
            mean_loss_epoch = np.mean(loss_epoch)
            acc = self.accuracy(torch.stack(y_pred).detach().cpu().float(),
                                torch.stack(y_true).detach().cpu())

        return mean_loss_epoch, acc 
