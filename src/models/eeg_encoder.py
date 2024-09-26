import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img")

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
                 n_channels: int = 128,
                 n_samples: int = 512,
                 n_classes: int = 40,
                 model_path: str = None,
                 **kwargs
                 ):
        super(EEGEncoder, self).__init__()

        device = kwargs["device"]

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.backbone_type = backbone
        self.checkpoint = torch.load(model_path)['model_state_dict'] if model_path is not None else None
        
        if self.checkpoint is not None:
            new_state_dict = {}
            for key, value in self.checkpoint.items():
                # Remove 'eeg_backbone.' from the keys
                new_key = key.replace("eeg_backbone.", "")
                new_state_dict[new_key] = value
            
            self.checkpoint = new_state_dict
        
        dropout_rate = 0.5
        kernel_length = 64
        f1 = 8
        d = 2
        f2 = 16

        if backbone == 'eegnet':
            self.eeg_backbone = eeg_architectures.EEGNet(n_samples=n_samples, n_channels=n_channels, n_classes=n_classes) 
            if self.checkpoint:
                self.eeg_backbone.load_state_dict(self.checkpoint) 
            self.feature_dim = list(self.eeg_backbone.model.children())[-1].in_features
            self.return_node = 'model.do2'
            # self.eeg_backbone = nn.Sequential(*list(self.eeg_backbone.model.children())[:-1])
            print(get_graph_node_names(self.eeg_backbone))

        elif backbone == 'EEGChannelNet':
            # out_channels = 32 for the sake of memory and 50 default, num_residual_blocks = 3 for Ahmed
            # and 4 for Spampinato
            self.eeg_backbone = eeg_architectures.EEGChannelNet(num_classes=n_classes, input_height=n_channels, input_width=n_samples, out_channels=32)
            if self.checkpoint:
                self.eeg_backbone.load_state_dict(self.checkpoint) 
            self.feature_dim = list(self.eeg_backbone.children())[-1][0].in_features
            self.return_node = 'view'
            print(get_graph_node_names(self.eeg_backbone))

        elif backbone == 'lstm':
            self.eeg_backbone = eeg_architectures.lstm(input_size=n_channels, lstm_size=kwargs['lstm_size'],
                                           lstm_layers=kwargs['lstm_layers'], device=device)
            print(get_graph_node_names(self.eeg_backbone))
        elif backbone == 'resnet1d':
            self.eeg_backbone = eeg_architectures.ResNet1d(
                n_channels=n_channels, 
                n_samples=n_samples, 
                net_filter_size=[16, 16, 32, 32, 64], 
                # net_seq_length=[n_samples, 128, 64, 32, 16],
                net_seq_length=[n_samples, 128, 64, 32, 16], 
                n_classes=n_classes)
            if self.checkpoint:
                self.eeg_backbone.load_state_dict(self.checkpoint) 
            self.feature_dim = list(self.eeg_backbone.children())[-1].in_features
            self.return_node = 'view'
            print(get_graph_node_names(self.eeg_backbone))
        else:
            raise NotImplementedError
        
        self.eeg_backbone = create_feature_extractor(self.eeg_backbone, return_nodes=[self.return_node])
        print("feature dim = ", self.feature_dim)
        # self.feature_dim = self.eeg_backbone(
        #     torch.zeros(1, 1, n_channels, n_samples)).contiguous().view(-1).size()[0]
        self.repr_layer = torch.nn.Linear(self.feature_dim, embed_dim)

    def forward(self, x):
    
        if self.backbone_type == "resnet1d":
            x = x.squeeze(1)
        out = self.eeg_backbone(x)[self.return_node]
        out = out.view(out.size(0), -1)
        embedding = self.repr_layer(out)
        
        return embedding.squeeze(dim=1), out.squeeze(dim=1)


if __name__ == "__main__":
    path_to_chkpnt = "/proj/rep-learning-robotics/users/x_nonra/data/asif_out/spampinato_eegnet_2024-09-16_11-21-04/eegnet_spampinato.pth"
    model = EEGEncoder(backbone="eegnet", model_path=path_to_chkpnt, device="cuda" if torch.cuda.is_available() else "cpu")
    x_in = torch.randn((1, 1, 128, 512))
    x_out = model(x_in)
    print(x_out.shape)

    