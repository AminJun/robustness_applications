import torchvision
import pdb
import torch as ch
from tqdm import tqdm
from dist import ImageNetMultiVariate
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT
from soft_xent import SoftCrossEntropy, CachedData
import sys
from user_constants import DATA_PATH_DICT

GLOBAL_MODE = int(sys.argv[1])
# 4: xent
# 3: MultiVarLabel
# 2: 1 Train
# 1: 1 Eval
# 0: Mean(Eval)

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

train_loader, test_loader = dataset.make_loaders(workers=NUM_WORKERS,
                                                 batch_size=BATCH_SIZE,
                                                 data_aug=False)


class FirstOutputWrapper(ch.nn.Module):
    def __init__(self, model: ch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, x) -> ch.tensor:
        return self.m(x)[0]


cached_data = CachedData('.', )
cached_data.cache(model, train_loader, test_loader)
label = cached_data()
print(label)


def get_loss(mode: int):
    def generation_loss(mod, inp, targ):
        op = mod(inp)
        # if mode is 4:
        #    loss = ch.nn.CrossEntropyLoss(reduction='none')(op, targ)
        # else:
        #     loss = SoftCrossEntropy(label, reduction='none')(op, targ)
        #    pdb.set_trace()
        loss = SoftCrossEntropy(label, reduction='none')(op, targ)
        return loss, None

    return generation_loss


kwargs = {
    'custom_loss': get_loss(GLOBAL_MODE),
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

print("========== Starting =============")
for i in tqdm(range(NUM_CLASSES_VIS)):
    target_class = i * ch.ones((BATCH_SIZE,)).cuda()
    im_seed = ch.stack([conditionals[int(t)].sample().view(3, DATA_SHAPE // GRAIN, DATA_SHAPE // GRAIN)
                        for t in target_class])

    im_seed = upsample(ch.clamp(im_seed, min=0, max=1)).cuda()
    _, im_gen = model(im_seed, target_class.long(), make_adv=True, **kwargs)
    images.append(im_gen)
torchvision.utils.save_image(ch.cat(images), f'desktop/comp{GLOBAL_MODE}.png')
