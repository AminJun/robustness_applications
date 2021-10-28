import os
import sys

from torch import nn
from torch.utils.data import DataLoader
import torch
from .base import SoftLabelData
import pdb


class CachedData(SoftLabelData):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _run_or_load(self, callable_func, *args, **kwargs):
        name = f'{callable_func.__name__}.pt'
        par = os.path.join(self._path, 'checkpoints', 'auto_save')
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
    def cache(self, model: nn.Module, train: DataLoader, eval: DataLoader):
        t_x, t_y = self._run_or_load(self.cache_train, model=model, train=train)
        e_x, e_y = self._run_or_load(self.cache_eval, model=model, eval=eval)
        pdb.set_trace()

    @torch.no_grad()
    def cache_train(self, model: nn.Module, train: DataLoader) -> (torch.tensor, torch.tensor):
        return self.cache_loader(model, train)

    @torch.no_grad()
    def cache_eval(self, model: nn.Module, eval: DataLoader) -> (torch.tensor, torch.tensor):
        return self.cache_loader(model, eval)

    @torch.no_grad()
    def cache_loader(self, model: nn.Module, loader: DataLoader) -> (torch.tensor, torch.tensor):
        images, labels = [], []
        for x, y in loader:
            x = x.to(self._device)
            y = y.to(self._device)
            pdb.set_trace()
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
