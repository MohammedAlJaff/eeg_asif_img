import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import model.training_utils as ut
import copy 


class UnimodalTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss,
                 save_path, filename, device='cuda', **kwargs):

        self.device = device

        self.model = model.to(device)
        self.loss = loss.to(device)
        self.loss_scaler = ut.NativeScalerWithGradNormCount()
        self.optimizer = optimizer

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
        scaler = GradScaler(enabled=self.mixed_precision)

        best_model = None
        best_loss = 10000000
        patience = self.patience
        print("Training Started...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}/{self.epochs}.")
            steps = 0
            loss_epoch = []
            self.model.train()
            progress_bar = tqdm(train_data_loader)
            for data, y in progress_bar:

                # adjust learning rate
                ut.adjust_learning_rate(self.optimizer, steps / len(train_data_loader) + epoch,
                                        warmup_epochs=self.warmup_epochs, num_epoch=self.epochs,
                                        lr=self.lr, min_lr=self.min_lr)

                x, _ = data

                self.optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)

                with autocast(enabled=self.mixed_precision):
                    preds = self.model(x)
                    loss = self.loss(preds, y)

                loss_epoch.append(loss.item())

                self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(), clip_grad=self.clip_grad)

                if self.device == torch.device('cuda'):
                    self.lr = self.optimizer.param_groups[0]["lr"]

                steps += 1

            train_loss = np.mean(loss_epoch)

            val_loss = self.evaluate(self.model, val_data_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = self.patience
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'loss': val_loss
                }
            else:
                patience -= 1

            if patience == 0:
                break

            print(f'Epoch: {epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}')

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss
            })

        print("Finished training.")
        print("Creating checkpoint.")

        if best_model is None:
            best_model = {
                'epoch': self.epochs,
                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                'loss': loss
            }

        print(f"Best Validation Loss = {best_model['loss']} (Epoch = {best_model['epoch']})")
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

            total_loss = []
            progress_bar = tqdm(dataloader)
            for data in progress_bar:
                eeg_samples = data.to(self.device)
                with autocast(enabled=self.mixed_precision):
                    loss, pred, _ = self.model(eeg_samples, mask_ratio=self.mask_ratio)
                loss_value = loss.item()

                pred = pred.to('cpu').detach()
                eeg_samples = eeg_samples.to('cpu').detach()
                pred = self.model_without_ddp.unpatchify(pred)

                total_loss.append(loss_value)

        mean_loss = np.mean(total_loss)

        return mean_loss, 
