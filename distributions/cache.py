import os
import torch
import sys


class CacheLocal:
    def __init__(self, path: str):
        self.par = os.path.join(path, 'checkpoints', 'auto_save', self.__class__.__name__)

    def run_or_load(self, callable_func, index: int = -1, *args, **kwargs):
        print(index)
        name = f'{callable_func.__name__}.pt' if index >= 0 else f'{callable_func.__name__}_{index}.pt'
        os.makedirs(self.par, exist_ok=True)
        path = os.path.join(self.par, name)
        if os.path.exists(path):
            print(f'{path} exists! Loading', file=sys.stderr)
            return torch.load(path)
        print(f'{path} not exists! Running', file=sys.stderr)
        result = callable_func(*args, **kwargs)
        torch.save(result, path)
        print(f'{path} Saved!', file=sys.stderr)
        return result
