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
    def binary_search_cache(dataset: Dataset):
        _, n_classes = dataset[len(dataset) - 1]
        n_classes = n_classes.item() + 1
        out = {k: (ClassSortedFactory._lb(dataset, k), ClassSortedFactory._up(dataset, k)) for k in range(n_classes)}

    @staticmethod
    def _get_label(dataset: Dataset, i: int) -> int:
        return dataset[i][1].item()

    @staticmethod
    def _lb(dataset: Dataset, label: int):
        f, e = 0, len(dataset)
        while f < e - 1:
            m = (f + e) // 2
            _, y = ClassSortedFactory._get_label(dataset, m)
            if y < label:
                f = m
            else:
                e = m
        _, y = dataset[f]
        assert y == label
        if f > 0:
            _, y = dataset[f - 1]
            assert y != label
        return f

    @staticmethod
    def _ub(dataset: Dataset, label: int):
        return ClassSortedFactory._lb(dataset, label + 1)

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
