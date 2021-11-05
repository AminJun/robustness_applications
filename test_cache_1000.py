import pdb

import torch
from torch.utils.data import DataLoader, Subset
import torchvision as tv

from datasets import image_net, ClassSortedFactory
from model import model_library
from distributions import CachedInits, CachedLabels


class IN1000RobustLabels(CachedLabels):
    pass


def main():
    model, image_size, batch_size, name = model_library[33]()
    inits = CachedInits('.', down_rate=8)

    classes = [10, 200, 980, 970, 37, 119, 281, 449]
    for c in classes:
        inits.cache(image_net, c)
    samples = []
    for _ in range(4):
        for c in classes:
            samples.append(inits(c))
    for _ in range(4):
        for c in classes:
            samples.append(inits(c, True))
    tv.utils.save_image(torch.cat(samples), 'sampled8.png', nrow=len(classes))

    factory_e = ClassSortedFactory(image_net, False, True)
    factory_t = ClassSortedFactory(image_net, True, True)
    eval = Subset(image_net.eval(), [i for c in classes for i in factory_e(c)])
    train_indices = [i for c in classes for i in factory_t(c)]
    pdb.set_trace()
    train = Subset(image_net.eval(), train_indices)
    eval = DataLoader(eval, batch_size, num_workers=4)
    train = DataLoader(train, batch_size, num_workers=4)

    labels_cache = IN1000RobustLabels('.')
    labels_cache.cache(model, train, eval)
    labels = []
    for i in range(7):
        labels.append(labels_cache(mode=i))
    pdb.set_trace()


if __name__ == '__main__':
    main()
