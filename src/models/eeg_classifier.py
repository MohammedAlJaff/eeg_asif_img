import sys

# sys.path.append("/proj/berzelius-2023-338/users/x_nonra/eeg_asif_img/")
sys.path.append("/mimer/NOBACKUP/groups/eeg_foundation_models/eeg_asif_img/")

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from src.models import eeg_architectures


class EEGClassifier(nn.Module):
    def __init__(self,
                 backbone="eegnet",
                 n_channels: int = 96,
                 n_samples: int = 512,
                 n_classes: int = 40,
                 **kwargs
                 ):
        super(EEGClassifier, self).__init__()

        device = kwargs["device"]

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.backbone_type = backbone

        dropout_rate = 0.5
        kernel_length = 64
        f1 = 8
        d = 2
        f2 = 16

        if backbone == 'eegnet':
            print('n_channels = ', n_channels)
            print('n_samples = ', n_samples)
            self.eeg_backbone = eeg_architectures.EEGNet(n_samples=n_samples, n_channels=n_channels, n_classes=n_classes)

        elif backbone == 'EEGChannelNet':
            # out_channels = 32 for the sake of memory and 50 default, num_residual_blocks = 3 for Ahmed
            # and 4 for Spampinato
            self.eeg_backbone = eeg_architectures.EEGChannelNet(num_classes=n_classes, input_height=n_channels, input_width=n_samples, out_channels=32)

        elif backbone == 'lstm':
            self.eeg_backbone = eeg_architectures.lstm(input_size=n_channels, lstm_size=kwargs['lstm_size'],
                                           lstm_layers=kwargs['lstm_layers'], n_classes=n_classes, device=device).to(device)

        elif backbone == 'resnet1d':
            self.eeg_backbone = eeg_architectures.ResNet1d(
                n_channels=n_channels, 
                n_samples=n_samples, 
                net_filter_size=[16, 16, 32, 32, 64], 
                net_seq_length=[n_samples, 128, 64, 32, 16], 
                n_classes=n_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
    
        if self.backbone_type == "resnet1d":
            x = x.squeeze(1)
        out = self.eeg_backbone(x)
        
        return out


if __name__ == "__main__":
    model_name = 'eegnet'
    model_configs = {
        'eegnet': {},
        'lstm': {'lstm_size': 128, 'lstm_layers': 1},
        'EEGChannelNet': {},
        'resnet1d': {}
    }
    model = EEGClassifier(
        backbone=model_name, 
        n_channels=128,
        n_samples=512,
        device="cuda" if torch.cuda.is_available() else "cpu", **model_configs[model_name])
    x_in = torch.randn((1, 1, 128, 512))
    x_out = model(x_in)
    print(x_out)