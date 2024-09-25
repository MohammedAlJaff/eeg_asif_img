import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ConvNextFeatureExtractor, ConvNextModel, DeiTImageProcessor, DeiTModel


class VIT():
    """
    Supervised image encoder based on the original ViT model from Google.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", device: str = None, load_jit=True):
        """
        Args:
            model_name: name of the pretrained model to use
            device: device to use for inference
            load_jit: load model just in time (in the encode method)
        """
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.load_jit = load_jit
        self.model_name = model_name
        if self.model_name == "google/vit-base-patch16-224-in21k":
            self.embedding_size = 768  # to be faster if embeddings are precomputed
        if not self.load_jit or self.model_name != "google/vit-base-patch16-224-in21k":
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name).to(self.device)
            self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768
        
    def encode(self, images: torch.Tensor, alr_preprocessed: bool = False) -> torch.Tensor:
        """
        Args:
            images: images to encode
            alr_preprocessed: whether the images are already preprocessed to tensors
        Returns:
            image embeddings
        """
        if self.load_jit:
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name).to(self.device)
            self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768
            self.load_jit = False  # we load the model a single time
        with torch.no_grad():
            if not alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                x = self.model(**x)
            else:
                x = self.model(images.to(self.device))
            x = x.last_hidden_state[:,0,:]
        return x  # taking the embedding of the CLS token as image representation


class DINO():
    """
    Unsupervised image encoder using the DINO model from Meta Research.
    """
    
    def __init__(self, model_name: str = 'facebook/dino-vits8', device: str = None, load_jit=True):
        """
        Args:
            model_name: name of the model to use: dino_vit{"s"mall or "b"ase}{"8" or "16" patch size}
            device: device to use.
            load_jit: load model just in time (in the encode method)
        """
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device 
        self.load_jit = load_jit
        self.model_name = model_name
        if self.model_name == 'facebook/dino-vits8':
            self.embedding_size = 384
        if not self.load_jit or self.model_name != 'facebook/dino-vits8':
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name, add_pooling_layer=False).to(self.device)
            self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768

    def encode(self, images: torch.Tensor, alr_preprocessed: bool = False) -> torch.Tensor:
        """
        Args:
            images: images to encode.
            alr_preprocessed: if True, the images are already preprocessed.
        Returns:
            encoded images.
        """
        if self.load_jit:
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name, add_pooling_layer=False).to(self.device)
            self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768
            self.load_jit = False  # we load the model a single time
        
        with torch.no_grad():
            if not alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                x = self.model(**x)
            else:
                x = self.model(images.to(self.device))
            x = x.last_hidden_state[:,0,:]
        return x  # taking the embedding of the CLS token as image representation


class DEIT():
    """
    Supervised image encoder based on the DEIT model from Meta (Visual Transformer).
    """
    def __init__(self, model_name: str = 'facebook/deit-tiny-distilled-patch16-224', device: str = None):
        """
        Args:
            model_name: name of the pretrained model to use
            device: device to use for inference
        """
        if not device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.feature_extractor = DeiTImageProcessor.from_pretrained(model_name)
        self.model = DeiTModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
        self.embedding_size = self.model(torch.zeros(1, 3, 224, 224).to(self.device)).last_hidden_state[:,0,:].shape[1]  # 768

    def encode(self, images: torch.Tensor, alr_preprocessed: bool = False) -> torch.Tensor:
        """
        Args:
            images: images to encode
            alr_preprocessed: whether the images are already preprocessed to tensors
        Returns:
            image embeddings
        """
        with torch.no_grad():
            if not alr_preprocessed:
                x = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                x = self.model(**x)
            else:
                x = self.model(images.to(self.device))
            x = x.last_hidden_state[:,0,:]
        return x