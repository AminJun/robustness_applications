import pdb

from torch import nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from .base import SoftLabelData
from ..cache import CacheLocal


class CachedLabels(SoftLabelData, CacheLocal):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, cache_path: str, mode: int = -1, n_classes: int = -1):
        SoftLabelData.__init__(self, n_classes=n_classes, mode=mode)
        CacheLocal.__init__(self, cache_path)
        self.one_t = self.one_e = self.mean_t = self.mean_e = self.sample_t = None
        self.sample_e = self.cov_e = self.cov_t = None
        self.sample_le = self.sample_lt = self.mean_lt = self.mean_le = None
        self.cov_lt = self.cov_le = None

    @torch.no_grad()
    def cache(self, model: nn.Module, train: DataLoader, test: DataLoader):
        t_x, t_y = self.run_or_load(self.cache_train, model=model, train=train)
        e_x, e_y = self.run_or_load(self.cache_test, model=model, test=test)
        # pdb.set_trace()

        sm = torch.nn.Softmax()
        t_l, e_l = t_x, e_x
        t_x, e_x = sm(t_x), sm(e_x)
        # pdb.set_trace()

        classes = t_y.unique()
        self.n = e_x.shape[-1]
        self.one_t = self.run_or_load(self.cache_one_train, predictions=t_x, targets=t_y, classes=classes)
        self.one_e = self.run_or_load(self.cache_one_test, predictions=e_x, targets=e_y, classes=classes)
        self.mean_t = self.run_or_load(self.cache_mean_t, predictions=t_x, targets=t_y, classes=classes)
        self.mean_e = self.run_or_load(self.cache_mean_e, predictions=e_x, targets=e_y, classes=classes)
        self.cov_t = self.run_or_load(self.cache_dist_t, p=t_x, t=t_y, m=self.mean_t, classes=classes)
        self.cov_e = self.run_or_load(self.cache_dist_e, p=e_x, t=e_y, m=self.mean_e, classes=classes)
        self.sample_t = self.run_or_load(self.cache_sampled_t, mean=self.mean_t, cov=self.cov_t, classes=classes)
        self.sample_e = self.run_or_load(self.cache_sampled_e, mean=self.mean_e, cov=self.cov_e, classes=classes)

        self.mean_lt = self.run_or_load(self.cache_mean_lt, predictions=t_l, targets=t_y, classes=classes)
        self.mean_le = self.run_or_load(self.cache_mean_le, predictions=e_l, targets=e_y, classes=classes)
        self.cov_lt = self.run_or_load(self.cache_dist_lt, p=t_l, t=t_y, m=self.mean_lt, classes=classes)
        self.cov_le = self.run_or_load(self.cache_dist_le, p=e_l, t=e_y, m=self.mean_le, classes=classes)
        self.sample_lt = self.run_or_load(self.cache_sampled_lt, mean=self.mean_lt, cov=self.cov_lt, classes=classes)
        self.sample_le = self.run_or_load(self.cache_sampled_le, mean=self.mean_le, cov=self.cov_le, classes=classes)

    @torch.no_grad()
    def cache_sampled_t(self, cov: torch.tensor, mean: torch.tensor,
                        classes: torch.tensor = None) -> torch.tensor:
        return self.cache_sampled(cov, mean, classes)

    @torch.no_grad()
    def cache_sampled_e(self, cov: torch.tensor, mean: torch.tensor,
                        classes: torch.tensor = None) -> torch.tensor:
        return self.cache_sampled(cov, mean, classes)

    @torch.no_grad()
    def cache_sampled_lt(self, cov: torch.tensor, mean: torch.tensor,
                         classes: torch.tensor = None) -> torch.tensor:
        return self.cache_sampled(cov, mean, classes)

    @torch.no_grad()
    def cache_sampled_le(self, cov: torch.tensor, mean: torch.tensor,
                         classes: torch.tensor = None) -> torch.tensor:
        return self.cache_sampled(cov, mean, classes)

    @torch.no_grad()
    def cache_sampled(self, cov: torch.tensor, mean: torch.tensor,
                      classes: torch.tensor = None) -> torch.tensor:
        pdb.set_trace()
        dists = {i.item(): MultivariateNormal(m, covariance_matrix=c) for m, c, i in zip(mean, cov, classes)}
        return torch.stack([dists[i.item()].sample() for i in classes])

    @torch.no_grad()
    def cache_dist_t(self, p: torch.tensor, t: torch.tensor, m: torch.tensor,
                     classes: torch.tensor = None) -> torch.tensor:
        return self.cache_dist(p, t, m, classes)

    @torch.no_grad()
    def cache_dist_e(self, p: torch.tensor, t: torch.tensor, m: torch.tensor,
                     classes: torch.tensor = None) -> torch.tensor:
        return self.cache_dist(p, t, m, classes)

    @torch.no_grad()
    def cache_dist_lt(self, p: torch.tensor, t: torch.tensor, m: torch.tensor,
                      classes: torch.tensor = None) -> torch.tensor:
        return self.cache_dist(p, t, m, classes)

    @torch.no_grad()
    def cache_dist_le(self, p: torch.tensor, t: torch.tensor, m: torch.tensor,
                      classes: torch.tensor = None) -> torch.tensor:
        return self.cache_dist(p, t, m, classes)

    @torch.no_grad()
    def cache_dist(self, p: torch.tensor, t: torch.tensor, m: torch.tensor, classes: torch.tensor = None
                   ) -> torch.tensor:
        cov = []

        for ii, i in enumerate(classes):
            imc = p[t == i]
            normalized = imc - m[ii].unsqueeze(dim=0)
            # pdb.set_trace()
            cov.append((normalized.t() @ normalized / len(imc)) + (1e-4 * torch.eye(self.n)))
        return torch.stack(cov)

    @torch.no_grad()
    def cache_one_train(self, predictions: torch.tensor, targets: torch.tensor, classes: torch.tensor = None
                        ) -> torch.tensor:
        return torch.stack([predictions[targets == i][0] for i in classes])

    @torch.no_grad()
    def cache_one_test(self, predictions: torch.tensor, targets: torch.tensor,
                       classes: torch.tensor = None) -> torch.tensor:
        return torch.stack([predictions[targets == i][0] for i in classes])

    @torch.no_grad()
    def cache_mean_t(self, predictions: torch.tensor, targets: torch.tensor,
                     classes: torch.tensor = None) -> torch.tensor:
        return self.cache_mean(predictions, targets, classes)

    @torch.no_grad()
    def cache_mean_e(self, predictions: torch.tensor, targets: torch.tensor,
                     classes: torch.tensor = None) -> torch.tensor:
        return self.cache_mean(predictions, targets, classes)

    @torch.no_grad()
    def cache_mean_lt(self, predictions: torch.tensor, targets: torch.tensor,
                      classes: torch.tensor = None) -> torch.tensor:
        return self.cache_mean(predictions, targets, classes)

    @torch.no_grad()
    def cache_mean_le(self, predictions: torch.tensor, targets: torch.tensor,
                      classes: torch.tensor = None) -> torch.tensor:
        return self.cache_mean(predictions, targets, classes)

    @torch.no_grad()
    def cache_mean(self, predictions: torch.tensor, targets: torch.tensor,
                   classes: torch.tensor = None) -> torch.tensor:
        return torch.stack([predictions[targets == i].mean(dim=0) for i in classes])

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
            # if len(images) > 100:
            #     break
        return torch.cat(images).cpu(), torch.cat(labels).cpu()

    def get_one_e(self, force_new: bool = True) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        return self.one_e

    def get_one_t(self, force_new: bool = True) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        return self.one_t

    def get_mean_t(self, force_new: bool = True) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        return self.mean_t

    def get_mean_e(self, force_new: bool = True) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        return self.mean_e

    def get_dist_t(self, force_new: bool = True) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        return self.sample_t

    def get_dist_e(self, force_new: bool = True) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        return self.sample_e

    def get_logit_t(self, force_new: bool = False) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        sm = torch.nn.Softmax()
        return sm(self.sample_lt)

    def get_logit_e(self, force_new: bool = False) -> torch.tensor:
        if force_new:
            raise NotImplementedError
        sm = torch.nn.Softmax()
        return sm(self.sample_le)
