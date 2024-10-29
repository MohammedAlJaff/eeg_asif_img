import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import math
from functools import partial


class EEGNet(nn.Module):
    def __init__(self, n_samples, n_channels=32, f1=8, d=2, f2=16, kernel_length=64, dropout_rate=0.5, n_classes=2):
        super().__init__()

        # implementation of the original EEGNet without the classification layer
        # Lawhern, Vernon J., et al. "EEGNet: a compact convolutional neural network for EEG-based brain–computer
        # interfaces." Journal of neural engineering 15.5 (2018): 056013.
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, kernel_length), padding='same')),
            ('bn1', nn.BatchNorm2d(f1)),
            ('conv2', nn.Conv2d(in_channels=f1, out_channels=d * f1, kernel_size=(n_channels, 1),
                                groups=f1, padding='valid')),
            ('bn2', nn.BatchNorm2d(d * f1)),
            # ('relu1', nn.ReLU()),
            ('elu1', nn.ELU()),  # original EEGNet
            ('pool1', nn.AvgPool2d(kernel_size=(1, 4))),
            ('do1', nn.Dropout(p=dropout_rate)),
            ('conv3', nn.Conv2d(in_channels=d * f1, out_channels=f2, kernel_size=(1, 16), groups=f2,
                                padding='same')),
            ('conv4', nn.Conv2d(in_channels=f2, out_channels=f2, kernel_size=1, padding='same')),
            ('bn3', nn.BatchNorm2d(f2)),
            # ('relu2', nn.ReLU()),
            ('elu2', nn.ELU()),  # original EEGNet
            ('pool2', nn.AvgPool2d(kernel_size=(1, 8))),
            ('do2', nn.Dropout(p=dropout_rate)),
            ('flat', nn.Flatten()),
            ('lnout', nn.Linear(f2 * (n_samples // 32), n_classes if n_classes > 2 else 1))
        ]))

    def forward(self, x):
        x = self.model(x)
        return x
        # return x.unsqueeze(1)


class lstm(nn.Module):
    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128, n_classes=2, device='cuda'):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, n_classes if n_classes > 2 else 1)
        self.device = device
    def forward(self, x):
        # Forward LSTM and get final state
        h_0 = torch.randn(self.lstm_layers, x.shape[0], self.lstm_size).to(self.device)
        c_0 = torch.randn(self.lstm_layers, x.shape[0], self.lstm_size).to(self.device)
        x = x.to(self.device)
        x = x.squeeze(dim=1).permute((0, 2, 1))
        x, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Forward output
        x = nn.functional.relu(self.output(hn[-1, ...]))
        x = self.classifier((x))

        return x

class subject_module(nn.Module):

    def __init__(self, subject_ids, n_filters_in, n_filters_out, kernel_size, downsample, padding):
        super(subject_module, self).__init__()
        
        # Create a dictionary where each subject has a combined Conv1d + BatchNorm1d module
        self.conv1 = nn.ModuleDict({
            str(subj_id): nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False, stride=downsample, padding=padding) for subj_id in subject_ids
        })

        self.bn1 = nn.BatchNorm1d(n_filters_out)

        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.padding = padding

    def forward(self, x, subj_id):
        
        # Apply the combined Conv1d + BatchNorm1d layer for the given subject ID
        if isinstance(subj_id, list):
            x = [self.conv1[str(id)](x_i) for id, x_i in zip(subj_id, x)]
            x = torch.stack(x)  # Stack back into a tensor after processing each element
        else:
            x = self.conv1[str(subj_id)](x)
        
        x = self.bn1(x)

        return x
    
    def add_subject(self, subj_id):
        # Check if the subject already exists
        if subj_id in self.conv1.keys():
            print(f"Subject {subj_id} already exists!")
        else:
            # Add a new Conv1d + BatchNorm1d module for the new subject
            self.conv1.update({str(subj_id): nn.Conv1d(self.n_filters_in, self.n_filters_out, self.kernel_size, bias=False, stride=self.downsample, padding=self.padding)})
            print(f"Subject {subj_id} added successfully!")


class BrainMLP(nn.Module):
    def __init__(self, out_dim=768, in_dim=1700, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        '''
            bs, 1, 15724 -> bs, 32, h
            bs, 32, h -> bs, 32h
            b2, 32h -> bs, 768
        '''
        # if x.ndim == 4:
        #     # case when we passed 3D data of shape [N, 81, 104, 83]
        #     assert x.shape[2] == 17 and x.shape[3] == 128
        #     # [N, 699192]
        x = x.reshape(x.shape[0], -1)

        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(x.shape[0], -1)
        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(x.shape[0], -1, self.clip_size))
        return x


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock_Subject(nn.Module):

    def __init__(self, subject_ids, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        super(ResBlock_Subject, self).__init__()

        self.res_blocks = nn.ModuleDict({
            str(subj_id): ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate) for subj_id in subject_ids
        })
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.kernel_size = kernel_size
        self.downsample = downsample

    def forward(self, x, y, subj_id):
        if isinstance(subj_id, list):
            tmp = [self.res_blocks[str(id)](x_i.unsqueeze(0), y_i.unsqueeze(0)) for id, x_i, y_i in zip(subj_id, x, y)]
            x = [tmp_i[0].squeeze(0) for tmp_i in tmp]
            y = [tmp_i[1].squeeze(0) for tmp_i in tmp]
            x = torch.stack(x)  # Stack back into a tensor after processing each element
            y = torch.stack(y)
        else:
            x, y = self.res_blocks[str(subj_id)](x, y)
        
        return x, y

    def add_subject(self, subj_id):
        # Check if the subject already exists
        if subj_id in self.res_blocks.keys():
            print(f"Subject {subj_id} already exists!")
        else:
            # Add a new Conv1d + BatchNorm1d module for the new subject
            self.res_blocks.update({str(subj_id): ResBlock1d(self.n_filters_in, self.n_filters_out, self.downsample, self.kernel_size, self.dropout_rate)})
            print(f"Subject {subj_id} added successfully!")


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_channels, n_samples, net_filter_size, net_seq_length, n_classes, kernel_size=17,
                 dropout_rate=0.5, **kwargs):
        super(ResNet1d, self).__init__()
        # my modifications!
        input_dim = (n_channels, n_samples)
        blocks_dim = list(zip(net_filter_size, net_seq_length))
        if n_classes == 2:
            n_classes = 1

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
class ResNet1d_Subject(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_channels, n_samples, net_filter_size, net_seq_length, n_classes, kernel_size=17,
                 dropout_rate=0.5, subject_ids={}, **kwargs):
        super(ResNet1d_Subject, self).__init__()
        # my modifications!
        input_dim = (n_channels, n_samples)
        blocks_dim = list(zip(net_filter_size, net_seq_length))
        if n_classes == 2:
            n_classes = 1

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.subj_spec_conv = subject_module(subject_ids=subject_ids, n_filters_in=n_filters_in, n_filters_out=n_filters_out, 
                                             kernel_size=kernel_size, downsample=downsample, padding=padding)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x, subj_id):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.subj_spec_conv(x, subj_id)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        # x = self.lin(x)
        return x


class ResNet1d_Subj_ResBlk(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_channels, n_samples, net_filter_size, net_seq_length, n_classes, kernel_size=17,
                 dropout_rate=0.5, subject_ids={}, **kwargs):
        super(ResNet1d_Subj_ResBlk, self).__init__()
        # my modifications!
        input_dim = (n_channels, n_samples)
        blocks_dim = list(zip(net_filter_size, net_seq_length))
        if n_classes == 2:
            n_classes = 1

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            if i == 4:
                resblk1d = ResBlock_Subject(subject_ids, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
                self.bn_extra = nn.BatchNorm1d(n_filters_out)
            else:
                resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x, subj_id):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for i, blk in enumerate(self.res_blocks):
            if i == 4:
                x, y = blk(x, y, subj_id)
                x = self.bn_extra(x) 
            else:
                x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        # x = self.lin(x)
        return x    
# The model proposed by the work: S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah,
# Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features,
# IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2
class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()

        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1)), 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)  # - 1 * (kernel[1] // 2 - 1)
            padding.append((temp_pad, 0))

        feature_height = input_height // stride[0]

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class EEGChannelNet_FE(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list,
            input_width
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1
                )
            ) for i in range(num_residual_blocks)
        ])

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def forward(self, x):
        out = self.temporal_block(x)

        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)
        # out = out.view(x.size(0), -1)

        return out

class EEGChannelNet(nn.Module):
    '''The model for EEG classification.
    The imput is a tensor where each row is a channel the recorded signal and each colums is a time sample.
    The model performs different 2D to extract temporal e spatial information.
    The output is a vector of classes where the maximum value is the predicted class.
    Args:
        in_channels: number of input channels
        temp_channels: number of features of temporal block
        out_channels: number of features before classification
        num_classes: number possible classes
        embedding_size: size of the embedding vector
        input_width: width of the input tensor (necessary to compute classifier input size)
        input_height: height of the input tensor (necessary to compute classifier input size)
        temporal_dilation_list: list of dilations for temporal convolutions, second term must be even
        temporal_kernel: size of the temporal kernel, second term must be even (default: (1, 32))
        temporal_stride: size of the temporal stride, control temporal output size (default: (1, 2))
        num_temp_layers: number of temporal block layers
        num_spatial_layers: number of spatial layers
        spatial_stride: size of the spatial stride
        num_residual_blocks: the number of residual blocks
        down_kernel: size of the bottleneck kernel
        down_stride: size of the bottleneck stride
        '''
    def __init__(self, in_channels=1, temp_channels=10, out_channels=50, num_classes=40, embedding_size=1000,
                 input_width=440, input_height=128, temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
                 temporal_kernel=(1,33), temporal_stride=(1,2),
                 num_temp_layers=4,
                 num_spatial_layers=4, spatial_stride=(2,1), num_residual_blocks=4, down_kernel=3, down_stride=2):
        super().__init__()

        self.encoder = EEGChannelNet_FE(in_channels, temp_channels, out_channels, input_width, input_height,
                                     temporal_kernel, temporal_stride,
                                     temporal_dilation_list, num_temp_layers,
                                     num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride
                                     )

        encoding_size = self.encoder(torch.zeros(1, in_channels, input_height, input_width)).contiguous().view(-1).size()[0]
        print(encoding_size)

        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, embedding_size),
            nn.ReLU(True),
            nn.Linear(embedding_size, num_classes if num_classes > 2 else 1), 
        )

    def forward(self, x):
        out = self.encoder(x)

        out = out.view(x.size(0), -1)

        out = self.classifier(out)

        return out