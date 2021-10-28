import os
import sys
import torch as ch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
from robustness.tools.label_maps import CLASS_DICT
from user_constants import DATA_PATH_DICT
from soft_xent.cache import CachedData

# Constants
DATA = 'RestrictedImageNet'  # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
BATCH_SIZE = 10
NUM_WORKERS = 8
NUM_CLASSES_VIS = 10

DATA_SHAPE = 32 if DATA == 'CIFAR' else 224  # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048  # Size of representation vector (fixed for model)
CLASSES = CLASS_DICT[DATA]  # Class names for dataset
NUM_CLASSES = len(CLASSES) - 1
NUM_CLASSES_VIS = min(NUM_CLASSES_VIS, NUM_CLASSES)
GRAIN = 4 if DATA != 'CIFAR' else 1

# Load dataset
dataset_function = getattr(datasets, DATA)
dataset = dataset_function(DATA_PATH_DICT[DATA])
train_loader, test_loader = dataset.make_loaders(workers=NUM_WORKERS,
                                                 batch_size=BATCH_SIZE,
                                                 data_aug=False)
data_iterator = enumerate(test_loader)

# Load model
model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': f'./models/{DATA}.pt'
}

model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()

c = CachedData('.', )
c.cache(model, train_loader, test_loader)


def downsample(x, step=GRAIN):
    down = ch.zeros([len(x), 3, DATA_SHAPE // step, DATA_SHAPE // step])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            v = x[:, :, i:i + step, j:j + step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii + 1, jj:jj + 1] = v
    return down


def upsample(x, step=GRAIN):
    up = ch.zeros([len(x), 3, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i + step, j:j + step] = x[:, :, ii:ii + 1, jj:jj + 1]
    return up


# Get seed distribution (can be memory intensive to do all ImageNet classes at once)

im_test, targ_test = [], []
for _, (im, targ) in enumerate(test_loader):
    im_test.append(im)
    targ_test.append(targ)
im_test, targ_test = ch.cat(im_test), ch.cat(targ_test)

conditionals = []
for i in tqdm(range(NUM_CLASSES_VIS)):
    imc = im_test[targ_test == i]
    down_flat = downsample(imc).view(len(imc), -1)
    mean = down_flat.mean(dim=0)
    down_flat = down_flat - mean.unsqueeze(dim=0)
    cov = down_flat.t() @ down_flat / len(imc)
    dist = MultivariateNormal(mean,
                              covariance_matrix=cov + 1e-4 * ch.eye(3 * DATA_SHAPE // GRAIN * DATA_SHAPE // GRAIN))
    conditionals.append(dist)

# Visualize seeds
img_seed = ch.stack([conditionals[i].sample().view(3, DATA_SHAPE // GRAIN, DATA_SHAPE // GRAIN)
                     for i in range(NUM_CLASSES_VIS)])
img_seed = ch.clamp(img_seed, min=0, max=1)
show_image_row([img_seed.cpu()], tlist=[[f'Class {i}' for i in range(NUM_CLASSES_VIS)]])
