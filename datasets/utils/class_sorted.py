import pdb

from torch.utils import model_zoo
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import EasyDataset


class ClassSortedFactory:
    def __init__(self, other: EasyDataset, train: bool, download: bool = False):
        self.other = other
        self.indices = {}
        name = f'{other.__class__.__name__}_{"eval" if not train else "eval"}'
        url = f'https://github.com/AminJun/PublicModels/releases/download/main/{name}.pt'
        if download:
            self.indices = model_zoo.load_url(url, map_location='cpu')
        else:
            self.indices = self.cache(other.eval() if train else other.train())

    @staticmethod
    def cache(dataset: Dataset) -> {}:
        out = {}
        for i, (_, y) in enumerate(tqdm(dataset)):
            if y not in out.keys():
                out[y] = []
            out[y].append(i)
            if len(out) > 10:
                break
        pdb.set_trace()
        return out

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
