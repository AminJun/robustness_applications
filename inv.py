import torchvision
import torch as ch
from tqdm import tqdm
from dist import ImageNetMultiVariate
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
from robustness.tools.label_maps import CLASS_DICT
from soft_cross_entorpy import SoftCrossEntropy
from user_constants import DATA_PATH_DICT

# Constants
DATA = 'RestrictedImageNet'
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

model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': f'./models/{DATA}.pt'
}

model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()


def upsample(x, step=GRAIN):
    up = ch.zeros([len(x), 3, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i + step, j:j + step] = x[:, :, ii:ii + 1, jj:jj + 1]
    return up


# Visualize seeds
conditionals = ImageNetMultiVariate()
img_seed = ch.stack([conditionals[i].sample().view(3, DATA_SHAPE // GRAIN, DATA_SHAPE // GRAIN)
                     for i in range(NUM_CLASSES_VIS)])
img_seed = ch.clamp(img_seed, min=0, max=1)
show_image_row([img_seed.cpu()], tlist=[[f'Class {i}' for i in range(NUM_CLASSES_VIS)]])


def get_loss(mode: int):
    def generation_loss(mod, inp, targ):
        op = mod(inp)
        if mode is 4:
            loss = ch.nn.CrossEntropyLoss(reduction='none')(op, targ)
        else:
            loss = SoftCrossEntropy(mode=mode)
        return loss, None

    return generation_loss


kwargs = {
    'custom_loss': get_loss(4),
    'constraint': '2',
    'eps': 40,
    'step_size': 1,
    'iterations': 60,
    'targeted': True,
}

if DATA == 'CIFAR':
    kwargs['eps'] = 30
    kwargs['step_size'] = 0.5
    kwargs['iterations'] = 60

show_seed = False
images = []
for i in tqdm(range(NUM_CLASSES_VIS)):
    target_class = i * ch.ones((BATCH_SIZE,)).cuda()
    im_seed = ch.stack([conditionals[int(t)].sample().view(3, DATA_SHAPE // GRAIN, DATA_SHAPE // GRAIN)
                        for t in target_class])

    im_seed = upsample(ch.clamp(im_seed, min=0, max=1)).cuda()
    _, im_gen = model(im_seed, target_class.long(), make_adv=True, **kwargs)
    images.append(im_gen)
torchvision.utils.save_image(ch.cat(images), 'desktop/gen_all.png')
