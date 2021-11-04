import torch
from torch.utils.data import DataLoader
import torchvision as tv

from datasets import image_net
from model import model_library
from distributions import CachedInits


def main():
    arc = model_library[33]
    model, image_size, batch_size, name = arc()
    loader = DataLoader(image_net.eval(), batch_size, shuffle=False)
    inits = CachedInits('.', down_rate=4)

    classes = [10, 200, 980, 970, 37, 119, 281, 449]
    for c in classes:
        inits.cache(loader, c)
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
