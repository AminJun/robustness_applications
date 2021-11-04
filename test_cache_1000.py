import torch
from torch.utils.data import DataLoader, Subset
import torchvision as tv

from datasets import image_net, ClassSortedFactory
from model import model_library
from distributions import CachedInits


def main():
    # arc = model_library[33]
    # model, image_size, batch_size, name = arc()

    inits = CachedInits('.', down_rate=4)

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
    tv.utils.save_image(torch.cat(samples), 'sampled.png', nrow=len(classes))


if __name__ == '__main__':
    main()
