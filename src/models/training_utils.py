import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import inf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
import os


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def adjust_learning_rate(optimizer, epoch, **kwargs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < kwargs['warmup_epochs']:
        lr = kwargs['lr'] * epoch / kwargs['warmup_epochs']
    else:
        lr = kwargs['min_lr'] + (kwargs['lr'] - kwargs['min_lr']) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - kwargs['warmup_epochs']) / (kwargs['num_epoch'] - kwargs['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


@torch.no_grad()
def plot_recon_figures(recon_model, curr_device, dataset, output_path, num_figures=5, config=None, logger=None,
                       model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    recon_model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30, 15))
    fig.tight_layout()
    axs[0, 0].set_title('Ground-truth')
    axs[0, 1].set_title('Masked Ground-truth')
    axs[0, 2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))
        sample = sample.to(curr_device)
        _, pred, mask = recon_model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask, mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_lr, start_lr=1e-6):
        """
        Warmup scheduler for the first `warmup_epochs`.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer for which to schedule the learning rate.
            warmup_epochs (int): Number of epochs to warm up the learning rate.
            max_lr (float): Maximum learning rate after warmup.
            start_lr (float): Starting learning rate at the beginning of warmup.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        """Update the learning rate for the current epoch."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linearly increase the learning rate during warmup
            lr = self.start_lr + (self.max_lr - self.start_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class CLIPLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        logit_scale = torch.exp(self.logit_scale)
        z_i_logits = (z_i @ z_j.T) * logit_scale
        z_j_logits = z_i_logits.T

        batch_size = z_i.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=z_i.device)

        loss_ij = nn.functional.cross_entropy(z_i_logits, labels)
        loss_ji = nn.functional.cross_entropy(z_j_logits, labels)

        return (loss_ij + loss_ji) / 2