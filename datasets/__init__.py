from .cifar import cifar10, cifar100
from .imagenet import image_net
from datasets.utils.normalizer import Normalizer
from .vit_imagenet import weird_image_net
from .utils import ClassSortedFactory, Normalizer

vision_datasets = {'cifar10': cifar10,
                   'cifar100': cifar100,
                   'imagenet': image_net,
                   'vit_imagenet': weird_image_net,
                   }
