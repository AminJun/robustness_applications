from torchvision.datasets import VisionDataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

from .base import EasyDataset


class ImageNet(EasyDataset):
    _root = './data/imagenet/{}'

    def __init__(self, img_size: int = 384):
        super(ImageNet, self).__init__(mean=[0.5], std=[0.5])

        self.eval_transforms = trans.Compose([trans.Resize(img_size), trans.CenterCrop(img_size), trans.ToTensor(), ])
        self.train_transforms = trans.Compose([trans.RandomResizedCrop(img_size), trans.RandomHorizontalFlip(),
                                               trans.ToTensor()])

    def eval(self) -> VisionDataset:
        return ImageFolder(root=self._root.format('val'), transform=self.eval_transforms)

    def train(self) -> VisionDataset:
        return ImageFolder(root=self._root.format('train'), transform=self.train_transforms)


weird_image_net = ImageNet()
