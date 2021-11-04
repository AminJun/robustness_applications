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

    factory = ClassSortedFactory(image_net, False, True)
    indices = [factory(c) for c in classes]
    indices = [i for li in indices for i in li]
    Subset(image_net.eval(), indices)
    labels_cache = IN1000RobustLabels('.')
    # labels_cache.cache(model, image)
    labels = []
    # for i in range(7):
    #     labels.append(labels_cache.)


if __name__ == '__main__':
    main()
