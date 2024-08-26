import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from src.models import eeg_architectures


class EEGEncoder(nn.Module): # TODO: now every architecture has a classification layer -> embedding should be extracted from penultimate layer.
    def __init__(self,
                 embed_dim=1024,
                 backbone="eegnet",
                 n_channels: int = 96,
                 n_samples: int = 512,
                 n_classes: int = 40,
                 **kwargs
                 ):
        super(EEGEncoder, self).__init__()

        device = kwargs["device"]

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.backbone_type = backbone

        dropout_rate = 0.5
        kernel_length = 64
        f1 = 8
        d = 2
        f2 = 16

        if backbone == 'eegnet':
            self.eeg_backbone = eeg_architectures.EEGNet(n_channels=n_channels)

            print(get_graph_node_names(self.eeg_backbone))

        elif backbone == 'EEGChannelNet':
            # out_channels = 32 for the sake of memory and 50 default, num_residual_blocks = 3 for Ahmed
            # and 4 for Spampinato
            self.eeg_backbone = eeg_architectures.EEGChannelNet(in_channels=1, temp_channels=10, out_channels=32,
                                                                input_width=n_samples, in_height=n_channels,
                                                                temporal_kernel=(1, 33), temporal_stride=(1, 2),
                                                                temporal_dilation_list=[(1, 1), (1, 2), (1, 4), (1, 8),
                                                                                        (1, 16)],
                                                                num_temporal_layers=4,
                                                                num_spatial_layers=4, spatial_stride=(2, 1),
                                                                num_residual_blocks=3, down_kernel=3, down_stride=2)
            print(get_graph_node_names(self.eeg_backbone))

        elif backbone == 'lstm':
            self.eeg_backbone = eeg_architectures.lstm(input_size=n_channels, lstm_size=kwargs['lstm_size'],
                                           lstm_layers=kwargs['lstm_layers'], device=device).to(device)
            # print(get_graph_node_names(self.eeg_backbone))
        elif backbone == 'resnet1d':
            self.eeg_backbone = eeg_architectures.ResNet1d(
                n_channels=n_channels, 
                n_samples=n_samples, 
                net_filter_size=[16, 16, 32, 32, 64], 
                net_seq_length=[n_samples, 128, 64, 32, 16], 
                n_classes=n_classes)
        else:
            raise NotImplementedError

        self.feature_dim = self.eeg_backbone(
            torch.zeros(1, 1, n_channels, n_samples)).contiguous().view(-1).size()[0]
        self.repr_layer = torch.nn.Linear(self.feature_dim, embed_dim)

    def forward(self, x):
    
        out = self.eeg_backbone(x)
        out = self.repr_layer(out)
        eeg_repr = out
        
        return out.squeeze(dim=1), eeg_repr


if __name__ == "__main__":
    model = EEGEncoder(backbone="eegnet", device="cuda:0" if torch.cuda.is_available() else "cpu")
    x_in = torch.randn((1, 1, 96, 512))
    x_out = model(x_in)