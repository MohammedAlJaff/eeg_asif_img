"""
Load data proposed by:
    C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, 
    Deep Learning Human Mind for Automated Visual Classification, International
    Conference on Computer Vision and Pattern Recognition, CVPR 2017
"""


import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from scipy.interpolate import interp1d
from PIL import Image
import pickle
import argparse
import yaml

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(n_px, interpolation=BICUBIC),
        T.CenterCrop(n_px),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-s', '--server', type=int, default=1)
    return parser.parse_args()


class EEGImagenet(Dataset):
    def __init__(
            self,
            subject_id,
            data_path,
            n_classes,
            z_score,
            img_size=224,
            fs=256,
            time_low=-0.1,
            time_high=1.0,
    ):

        self.z_score = z_score
        self.fs = fs
        self.eeg_data = None
        self.labels = None
        self.img_size = img_size
        self.time_low = int((time_low - 0.02) * self.fs)
        self.time_high = int((time_high - 0.02) * self.fs)
        print(f"image size = {img_size}")
        self.transforms = T.ToTensor()
        self.transforms_img = T.Compose([T.ToTensor(), T.ConvertImageDtype(dtype=torch.float32)])
        self.transforms_tensor2img = T.Compose([T.ToPILImage()])
        self.img_preprocess = _transform(img_size)

        self.labels = np.load(os.path.join(data_path, f"spampinato_et_al_label_{subject_id}.npy"), mmap_mode='r')
        self.eeg_data = np.load(os.path.join(data_path, f"spampinato_et_al_eeg_{subject_id}.npy"), mmap_mode='r')
        self.image_files = np.load(os.path.join(data_path, f"spampinato_et_al_img_{subject_id}.npy"), mmap_mode='r')
        self.pairs = np.load(data_path, f"images.npy", mmap_mode='r')
        self.subjects = np.zeros((len(self.eeg_data),))

        max_classes = 40

        if n_classes < max_classes:
            sample_classes = sorted(random.sample(range(0, max_classes), n_classes))
            self.class_dict = {str(y): x for x, y in enumerate(sample_classes)}  # change str labels to indices
            print(f"sample_classes = {sample_classes}")
            self.indices = [i for i, label in enumerate(self.labels) if label in sample_classes]
        else:
            self.indices = list(range(0, len(self.labels)))
            self.class_dict = {str(y): y for y in range(0, n_classes)}
            print(f"training on classes: all 40 classes")
        print(f"class_dict: {self.class_dict}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        eeg = self.eeg_data[self.indices[item]]
        eeg = eeg.copy()
        if eeg.shape[0] > 1:
            eeg = np.expand_dims(eeg, axis=0)
        eeg = eeg[:, :, self.time_low:self.time_high]

        # set time length to 512
        if eeg.shape[-1] > 512:
            idx = np.random.randint(0, int(eeg.shape[-1] - 512) + 1)

            eeg = eeg[..., idx: idx + 512]
        else:
            x1 = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, 512)
            f = interp1d(x1, eeg, axis=-1)
            eeg = f(x2)

        if self.z_score:
            # z_score normalization
            eeg = (eeg - np.mean(eeg, axis=-1, keepdims=True)) / (np.std(eeg, axis=-1, keepdims=True) + 1e-08)

        pair = self.pairs[self.indices[item]].copy()
        sample = (torch.from_numpy(eeg).to(torch.float), (self.img_preprocess(pair).to(torch.float)))
        label = self.class_dict[str(self.labels[self.indices[item]])]
        # img_file = self.image_files[self.indices[item]].copy()

        return sample, label


class Splitter(Dataset):
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        sample, label, img_file = self.dataset[self.split_idx[i]]
        # Return
        return sample, label, img_file


if __name__ == "__main__":
    selected = None
    eeg_path = "/Users/nonarajabi/Desktop/KTH/Projects/EEG_Data/eeg2img/spampinato_et_al"
    data_loaded = EEGImagenet(
        subject_id=0,
        data_path=eeg_path,
        n_classes=40,
        img_size=224,
        fs=1000,
        time_low=0.02,
        time_high=0.46,
    )
    print(len(data_loaded))
    x, l, im = data_loaded[0]
