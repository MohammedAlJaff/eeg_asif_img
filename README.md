This text contains instructions on how to run the code for the project. The code is written in Python and uses the version 3.9.16. 

# Container

An Apptainer image is provided to ensure a consistent and reproducible environment for running the code across different systems. The Apptainer image is available as *eeg_torch_container.sif*. You can download it from the following link: https://kth-my.sharepoint.com/:f:/g/personal/nonar_ug_kth_se/EhM240MftCVNkBVnnjPQNL4BnypPOuK4Hm4p-k9AXysVuw?e=ejLp6t 

# Dataset
The following datasets are available:
- Things-EEG-2
- ...

### [Things-EEG-2](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)
This dataset contains EEG recordings of human participants while observing natural images from the THINGS image database. All images come from THINGS ([Hebart et al., 2019](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792)), a database of 12 or more images of objects on a natural background for each of 1854 object concepts, where each concept (e.g., antelope, strawberry, t-shirt) belongs to one of 27 higher-level categories (e.g., animal, food, clothing). 

The 1854 object concepts has been pseudo-randomly divided into non-overlapping 1654 training and 200 test concepts under the constraint that the same proportion of the 27 higher-level categories had to be kept in both partitions. Ten images has been selected for each training partition concept and one image for each test partition concept, resulting in a training image partition of 16,540 image conditions (1654 training object concepts × 10 images per concept = 16,540 training image conditions) and a test image partition of 200 image conditions (200 test object concepts × 1 image per concept = 200 test image conditions). Each training image condition has been presented 4 times and each test image condition has been presented 80 times in total. In the current version of the code, we average the EEG data across the 4 repetitions for each training image condition and across the 80 repetitions for each test image condition.

The current version of our code uses the preprocessed EEG version of the dataset. This version contains EEG recordings from 200 ms before stimulus onset to 800 ms after stimulus onset. The EEG data has been downsampled to 100 Hz and 17 channels have been selected overlaying the occipital and parietal cortex. 

The EEG data for each participant is stored in a numpy array with the shape (n_trials, n_channels, n_timepoints), where n_trials is the number of trials, n_channels is the number of EEG channels, and n_timepoints is the number of timepoints. Therefore, the shape of the EEG data for each participant is (16540, 17, 100) and (200, 17, 100) for training and test sets respectively. 

## Usage

All datasets are implemented as torch.utils.data.Dataset classes in `src/datasets/eeg_image.py`. To download the Things-EEG-2 dataset, you can simply run the following code in the container:
    
    python src/datasets/eeg_image.py --dataset things-eeg-2 --data_path path/to/directory/where/data/is/saved --download

or outside the container:

    apptainer exec path/to/container/eeg_torch_container.sif python src/datasets/eeg_image.py --dataset things-eeg-2 --data_path path/to/directory/where/data/is/saved --download 

A storage size of ~10.5 GB is required to store the dataset. Don't forget to replace the `sys.path.append("/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img")` (line 3 in `eeg_image.py`) with the path to the repository on your local machine.

Unfortunately, this script only downloads the preprocessed data and the raw data is not included. You can manually download the raw data from [here](https://osf.io/crxs4/).

# EEG-Image Align

We want to train an EEG encoder that enables decoding the observed image stimulus from corresponding EEG data. For this we take a pretrained image encoder such as the OpeAI's CLIP image encoder and train the EEG encoder such that the final output EEG embeddings maximally align with the corresponding image embeddings from the image encoder. Note that we only train the EEG encoder and the image encoder is frozen during the whole training. 

In the current version we use:
- ResNet-1D as the *EEG encoder*
- CLIP-ViT-B/32 as the *image encoder*
- CLIP loss as the loss function (bidirectional cosine similarity between EEG and image embeddings)

All the models and training modules are available under `src/models/`. Specifically, one can find EEG backbone architectures under `src/models/eeg_architectures.py`. These architectures are used in `src/models/eeg_encoder.py` where a linear layer will be added on top of them to map their embedding space to the image embedding space.

BimodalTrainer in `src/models/traner.py` is used to run the training and validation loop.

## Main

The whole training and evaluation of the models happen in `src/eeg_img_align.py`. The input arguments that we currently care about are: 

- *--data_path*: path to the directory that contains all the datasets (without the specific dataset name) e.g.: /proj/rep-learning-robotics/users/x_nonra/eeg_asif_img/data
- *--save_path*: path to the directory to save the model
- *--checkpoint*: path to the pretrained model, only set it if you want to finetune a pretrained model otherwise leave it as default (None).
- *--dataset*: things-eeg-2.
- *--subject_id*: specifies the subjects whose test data should be used to train the EEG encoder (can be more than one).
- *--test_subject*: specifies the subject whose test data will be used to evaluate the model (only one subject can be specified). If not specified: test data will be used from the previous subject(s) defined by *subject_id*.
- *--subj_training_ratio*: a ratio between 0 and 1 determining how much of participants training samples to be used
- *--n_classes*: 1654 
- *--eeg_enc*: the EEG encoder architecture. We use *resnet1d*
- *--img_enc*: the image encoder architecture. We use *CLIP_IMG* which will be translated to CLIP-ViT-B/32
- *--loss*: specifies the loss function, for now only *clip-loss* is working.
- *--downstream*: the downstream task. Use *retrieval*. If not specified, no downstream task will be performed.
- *--epoch*: number of epochs (I used 300 for pretraining)
- *--finetune_epoch*: number of finetuning epochs on the downstream task if the task is classification (We actually don't use this)
- *--warmup*: The number of epochs in the beginning of training takes for the learning rate to reach from min_lr to lr
- *--temperature*: contrastive loss temperature 0.04
- *--n_workers*: number of workers for dataloaders (in Berzelius each CPU can handle 2 workers-I'm not sure if this is generally true about other computers)
- *--separate_test*: specifies if the separate test dataset (unseen 200 classes) should be used to evaluate the model. Otherwise, the test set will be sampled from the training dataset.
- *-b*: batch size (I used 512)
- *--lr*: learning rate (I used 0.005 for pretraining and 0.0005 for finetuning on the target subjct)
- *--seed*: I used 42


### Examples

Pretrain the model on data from all subjects except subject 10:

    apptainer exec --nv $CONTAINER python src/eeg_img_align.py --data_path "$data_path" --save_path "$save_path" --separate_test \
    --dataset things-eeg-2 --subject_id 1 2 3 4 5 6 7 8 9 --test_subject 10 --n_classes 1654 --eeg_enc "resnet1d" --img_enc "CLIP_IMG" --epoch 300 --modality "eeg-img" \
    -b 512 --n_workers 8 --lr 0.005 --warmup 20 --seed 42 --temperature 0.04

Finetune the pretrained model on subject 10. 

```
apptainer exec --nv $CONTAINER python src/eeg_img_align.py --data_path "$data_path" --save_path "$save_path" --split_path "$split_path" --checkpoint "$checkpoint" --separate_test \
--dataset things-eeg-2 --subject_id 10 --n_classes 1654 --eeg_enc "resnet1d" --img_enc "CLIP_IMG" --epoch 30 --modality "eeg-img" \
--downstream "retrieval" -b 512 --n_workers 8 --lr 0.0005 --warmup 5 --seed 42 --temperature 0.04 --subj_training_ratio "$tr"
```

