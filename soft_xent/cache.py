import os
import sys

from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from .base import SoftLabelData
import pdb


class CachedData(SoftLabelData):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _run_or_load(self, callable_func, *args, **kwargs):
        name = f'{callable_func.__name__}.pt'
        par = os.path.join(self._path, 'checkpoints', 'auto_save', self.__class__.__name__)
        os.makedirs(par, exist_ok=True)
        path = os.path.join(par, name)
        if os.path.exists(path):
            print(f'{path} exists! Loading', file=sys.stderr)
            return torch.load(path)
        print(f'{path} not exists! Running', file=sys.stderr)
        result = callable_func(*args, **kwargs)
        torch.save(result, path)
        print(f'{path} Saved!')
        return result

    def __init__(self, cache_path: str, mode: int = -1):
        super().__init__(-1, mode=mode)
        self._path = cache_path

    @torch.no_grad()
    def cache(self, model: nn.Module, train: DataLoader, test: DataLoader):
        t_x, t_y = self._run_or_load(self.cache_train, model=model, train=train)
        e_x, e_y = self._run_or_load(self.cache_test, model=model, test=test)

        sm = torch.nn.Softmax()
        t_x, e_x = sm(t_x), sm(e_x)

        classes = t_y.unique()
        self.n = len(classes)
        self.one_t = self._run_or_load(self.cache_one_train, predictions=t_x, targets=t_y)
        self.one_e = self._run_or_load(self.cache_one_test, predictions=e_x, targets=e_y)
        self.mean_t = self._run_or_load(self.cache_mean_t, predictions=t_x, targets=t_y)
        self.mean_e = self._run_or_load(self.cache_mean_e, predictions=e_x, targets=e_y)
        self.dist_e = self._run_or_load(self.cache_dist_t)
        self.cov_t = self._run_or_load(self.cache_dist_t, p=t_x, t=t_y)

    @torch.no_grad()
    def cache_dist_t(self, p: torch.tensor, t: torch.tensor) -> torch.tensor:
        return self.cache_dist(p, t)

    @torch.no_grad()
    def cache_dist(self, p: torch.tensor, t: torch.tensor) -> torch.tensor:
        pdb.set_trace()
        return None

    @torch.no_grad()
    def cache_one_train(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        return torch.stack([predictions[targets == i][0] for i in range(self.n)])

    @torch.no_grad()
    def cache_one_test(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        return torch.stack([predictions[targets == i][0] for i in range(self.n)])

    @torch.no_grad()
    def cache_mean_t(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        return self.cache_mean(predictions, targets)

    @torch.no_grad()
    def cache_mean_e(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        return self.cache_mean(predictions, targets)

    @torch.no_grad()
    def cache_mean(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        return torch.stack([predictions[targets == i].mean(dim=0) for i in range(self.n)])

    @torch.no_grad()
    def cache_train(self, model: nn.Module, train: DataLoader) -> (torch.tensor, torch.tensor):
        return self.cache_loader(model, train)

    @torch.no_grad()
    def cache_test(self, model: nn.Module, test: DataLoader) -> (torch.tensor, torch.tensor):
        return self.cache_loader(model, test)

    @torch.no_grad()
    def cache_loader(self, model: nn.Module, loader: DataLoader) -> (torch.tensor, torch.tensor):
        images, labels = [], []
        for x, y in tqdm(loader):
            x = x.to(self._device)
            y = y.to(self._device)
            images.append(model(x).detach().clone())
            labels.append(y.detach().clone())
        return torch.cat(images).cpu(), torch.cat(labels).cpu()

    def get_1_eval(self) -> torch.tensor:
        pass

    def get_1_train(self) -> torch.tensor:
        pass

    def get_mean(self) -> torch.tensor:
        pass

    def get_dist(self) -> torch.tensor:
        pass
