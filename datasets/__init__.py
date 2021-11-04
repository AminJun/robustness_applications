from .cifar import cifar10, cifar100
from .imagenet import image_net
from datasets.tools.normalizer import Normalizer
from .vit_imagenet import weird_image_net
from .tools import ClassSortedFactory, Normalizer

vision_datasets = {'cifar10': cifar10,
                   'cifar100': cifar100,
                   'imagenet': image_net,
                   'vit_imagenet': weird_image_net,
                   }
