import os
import pdb

import torch
import torchvision
from robustness.attacker import AttackerModel
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision as tv
from tqdm import tqdm

from datasets import image_net, ClassSortedFactory
from model import model_library
from distributions import CachedInits, CachedLabels, SoftCrossEntropy
from user_constants import DATA_PATH_DICT
from utils import exp_starter_pack


class IN1000RobustLabels(CachedLabels):
    pass


class LabelCompleter:
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, n_classes: int, classes: list):
        self.n = n_classes
        self.classes = sorted(classes)

    def __call__(self, label: torch.tensor) -> torch.tensor:
        output = label
        if label.size(0) != self.n:
            output = torch.eye(self.n).to(self._device)
            for l, c in zip(label, self.classes):
                output[c] = l
        return output.detach().clone()


def main():
    exp_name, args, _ = exp_starter_pack()
    method = args.method

    model, image_size, batch_size, name = model_library[33]()
    inits = CachedInits('.', down_rate=8)

    classes = [10, 200, 980, 970, 37, 119, 281, 449]
    for c in classes:
        inits.cache(image_net, c)
    samples = []
    # for _ in range(4):
    #     for c in classes:
    #         samples.append(inits(c))
    for _ in range(4):
        for c in classes:
            samples.append(inits(c, True))
    tv.utils.save_image(torch.cat(samples), 'sampled8.png', nrow=len(classes))

    factory_e = ClassSortedFactory(image_net, False, True)
    factory_t = ClassSortedFactory(image_net, True, True)
    eval = Subset(image_net.eval(), [i for c in classes for i in factory_e(c)])
    train_indices = [i for c in classes for i in factory_t(c)]
    train = Subset(image_net.train(), train_indices)
    eval = DataLoader(eval, batch_size, num_workers=4)
    train = DataLoader(train, batch_size, num_workers=4)

    labels_cache = IN1000RobustLabels('.', n_classes=1000)
    labels_cache.cache(model, train, eval)
    labels = []
    for i in range(7):
        labels.append(labels_cache(mode=i))

    comp = LabelCompleter(1000, classes)
    labels = [comp(l) for l in labels]
    label = labels[method]

    class IgnorantModel(nn.Module):
        def __init__(self, sub: nn.Module):
            super().__init__()
            self.sub = sub

        def forward(self, x: torch.tensor, *args, **kwargs):
            return self.sub(x)

    def generation_loss(mod, inp, targ):
        op = mod(inp)
        loss = SoftCrossEntropy(label, reduction='none')(op, targ)
        return loss, None

    kwargs = {
        'custom_loss': generation_loss,
        'constraint': '2',
        'eps': 40,
        'step_size': 1,
        'iterations': 60,
        'targeted': True,
    }

    images = []

    print("========== Starting =============")
    num_to_draw = 10
    BATCH_SIZE = 10

    DATA = 'ImageNet'
    from robustness import datasets
    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function(DATA_PATH_DICT[DATA])

    model = IgnorantModel(model.cuda()).cuda()
    model = AttackerModel(model, dataset).cuda()

    for i in tqdm(classes):
        target_class = i * torch.ones((BATCH_SIZE,)).long().cuda()
        im_seed = torch.cat([inits(t.item(), force_new=True) for t in target_class])

        im_seed = torch.clamp(im_seed, min=0, max=1).cuda()
        torchvision.utils.save_image(im_seed, 'before.png')
        _, im_gen = model(im_seed, target_class.long(), make_adv=True, attacker_kwargs={'do_tqdm': True}, **kwargs)
        torchvision.utils.save_image(im_seed, 'after.png')
        images.append(im_gen)

    images = torch.cat(images)
    os.makedirs(f'desktop/im1000_{method}', exist_ok=True)
    for i, im in enumerate(images):
        torchvision.utils.save_image(im, f'desktop/im1000_{method}/{i}.png')


if __name__ == '__main__':
    main()
