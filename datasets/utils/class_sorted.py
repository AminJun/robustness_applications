import pdb

import torch
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .base import EasyDataset


class ClassSortedFactory:
    def __init__(self, other: EasyDataset, train: bool, download: bool = False):
        self.other = other
        self.indices = {}
        self.name = f'{other.__class__.__name__}_{"eval" if not train else "eval"}'
        url = f'https://github.com/AminJun/PublicModels/releases/download/main/{self.name}.pt'
        if download:
            self.indices = model_zoo.load_url(url, map_location='cpu')
        else:
            self.indices = self.cache(other.train() if train else other.eval())
            self.save()

    def save(self):
        torch.save(self.indices, self.name)

    @staticmethod
    def cache(dataset: Dataset, batch_size: int = 100) -> {}:
        out = {}
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for ii, (_, yy) in enumerate(tqdm(loader)):
            for i, y in enumerate(yy):
                y = y.item()
                if y not in out.keys():
                    out[y] = []
                out[y].append(i + ii * batch_size)
        return {k: ClassSortedFactory.zip(v) for k, v in out.items()}

    @staticmethod
    def zip(indices: list) -> []:
        if len(indices) == 0:
            return []
        out: [tuple] = [(indices[0], indices[0] + 1)]
        for i in range(1, len(indices)):
            item = indices[i]
            if item == out[-1][1]:
                out[-1] = (out[-1][0], item + 1)
            else:
                out.append((item, item + 1))
        return out

    @staticmethod
    def unzip(indices: list) -> []:
        out = []
        for s, e in indices:
            for i in range(s, e):
                out.append(i)
        return out

    def __call__(self, label: int) -> list:
        return self.unzip(self.indices[label])
