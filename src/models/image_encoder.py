import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img")

import torch.nn as nn

from src.models.image_architectures import VIT, DINO, DEIT

class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "ViT",
        embed_dim: int = None,
        add_ln_layer: bool = False,
        **kwargs
        ):
        super().__init__()
        self.backbone = backbone
        if backbone == "ViT":
            self.image_backbone = VIT()
        elif backbone == "DINO":
            self.image_backbone = DINO()
        elif backbone == "DeiT":
            self.image_backbone = DEIT()
        else:
            raise NotImplementedError

        if add_ln_layer:
            assert embed_dim is not None
            self.fc = nn.Linear(self.image_backbone.embedding_size, embed_dim)
            self.embed_dim = embed_dim
        else:
            self.fc = None
            self.embed_dim = self.image_backbone.embedding_size
        print("image embedding size = ", self.embed_dim)

    def forward(self, x):
        x = self.image_backbone.encode(x)
        if self.fc is not None:
            x = self.fc(x)
        return x
