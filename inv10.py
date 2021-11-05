import os
import pdb

import torch
import torchvision
from robustness import model_utils
from robustness.attacker import AttackerModel
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision as tv
from tqdm import tqdm

from datasets import image_net, ClassSortedFactory
from dist import ImageNetMultiVariate
from model import model_library
from distributions import CachedInits, CachedLabels, SoftCrossEntropy
from user_constants import DATA_PATH_DICT
from utils import exp_starter_pack


# class IN1000RobustLabels(CachedLabels):
#     pass


# class LabelCompleter:
#     _device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     def __init__(self, n_classes: int, classes: list):
#         self.n = n_classes
#         self.classes = sorted(classes)
#
#     def __call__(self, label: torch.tensor) -> torch.tensor:
#         output = label
#         if label.size(0) != self.n:
#             output = torch.eye(self.n).to(self._device)
#             for l, c in zip(label, self.classes):
#                 output[c] = l
#         return output.detach().clone()


def main():
    exp_name, args, _ = exp_starter_pack()
    method = args.method

    # model, image_size, batch_size, name = model_library[33]()
    inits = ImageNetMultiVariate()
    batch_size = 10

    # inits = CachedInits('.', down_rate=4)

    # classes = [10, 200, 980, 970, 37, 119, 281, 449]
    # for c in classes:
    #     inits.cache(image_net, c)
    # samples = []
    # for _ in range(4):
    #     for c in classes:
    #         samples.append(inits(c))
    # for _ in range(4):
    #     for c in classes:
    #         samples.append(inits(c, True))
    # tv.utils.save_image(torch.cat(samples), 'sampled8.png', nrow=len(classes))

    # factory_e = ClassSortedFactory(image_net, False, True)
    # factory_t = ClassSortedFactory(image_net, True, True)
    # eval = Subset(image_net.eval(), [i for c in classes for i in factory_e(c)])
    # train_indices = [i for c in classes for i in factory_t(c)]
    # train = Subset(image_net.train(), train_indices)
    # eval = DataLoader(eval, batch_size, num_workers=4)
    # train = DataLoader(train, batch_size, num_workers=4)

    DATA = 'RestrictedImageNet'
    from robustness import datasets
    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function(DATA_PATH_DICT[DATA])
    train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=batch_size, data_aug=False)

    class FirstOutputWrapper(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.m = model

        def forward(self, x) -> torch.tensor:
            return self.m(x)[0]

    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./models/{DATA}.pt'
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()

    cached_data = CachedLabels('.', method)
    cached_data.cache(FirstOutputWrapper(model), train_loader, test_loader)
    label = cached_data().cuda()
    print(label.shape)
    print(label)
    # labels_cache = IN1000RobustLabels('.', n_classes=1000)
    # labels_cache.cache(model, train, eval)
    # labels = []
    # for i in range(7):
    #     labels.append(labels_cache(mode=i))
    #
    # comp = LabelCompleter(1000, classes)
    # labels = [comp(l) for l in labels]
    # label = labels[method]

    # class IgnorantModel(nn.Module):
    #     def __init__(self, sub: nn.Module):
    #         super().__init__()
    #         self.sub = sub
    #
    #     def forward(self, x: torch.tensor, *args, **kwargs):
    #         return self.sub(x)
    #
    def generation_loss(mod, inp, targ):
        op = mod(inp)
        loss = SoftCrossEntropy(label, reduction='none')(op, targ)
        return loss, None

    kwargs = {
        'custom_loss': generation_loss,
        'constraint': '2',
        'eps': 40 * args.l_norm,
        'step_size': 0.1,
        'iterations': 160,
        'targeted': True,
    }

    images = []

    print("========== Starting =============")
    num_to_draw = 10
    BATCH_SIZE = 8

    # DATA = 'ImageNet'
    # from robustness import datasets
    # dataset_function = getattr(datasets, DATA)
    # dataset = dataset_function(DATA_PATH_DICT[DATA])

    # model, _ = model_utils.make_and_restore_model(**model_kwargs)
    # model.eval()

    # model = IgnorantModel(model.cuda()).cuda()
    # model = AttackerModel(model, dataset).cuda()
    import numpy as np
    classes = np.arange(9)
    t_classes = []

    up = torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    for i in tqdm(classes):
        target_class = i * torch.ones((BATCH_SIZE,)).long().cuda()
        im_seed = torch.cat([up(inits[t.item()].sample().view(1, 3, 56, 56)) for t in target_class])
        # pdb.set_trace()
        t_classes.append(target_class)

        im_seed = torch.clamp(im_seed, min=0, max=1).cuda()
        _, im_gen = model(im_seed, target_class.long(), make_adv=True, do_tqdm=True, **kwargs)
        images.append(im_gen)

    images = torch.cat(images)
    t_classes = torch.cat(t_classes)
    os.makedirs(f'desktop/im1000_{method}_{args.l_norm}', exist_ok=True)
    for i, (im, t) in enumerate(zip(images, t_classes)):
        torchvision.utils.save_image(im, f'desktop/im10_{method}_{args.l_norm}/{t}_{i % BATCH_SIZE}.png')
    torchvision.utils.save_image(images, f'desktop/after_{method}.png')


if __name__ == '__main__':
    main()
