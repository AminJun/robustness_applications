import pdb

import torch
from torch.utils.data import DataLoader
import torchvision as tv

from datasets import image_net, ClassSortedFactory
from model import model_library
from distributions import CachedInits


def main():
    factory = ClassSortedFactory(image_net, False, True)
    pdb.set_trace()


if __name__ == '__main__':
    main()
