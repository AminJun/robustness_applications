import pdb

from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import ClassSortedFactory
from datasets.utils.base import EasyDataset
from ..cache import CacheLocal
import torch


class CachedInits(CacheLocal):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, path: str, down_rate: int = 1):
        super().__init__(path)
        self.par = f'{self.par}_{down_rate}'
        self.mean = {}
        self.cov = {}
        self.sample = {}
        self.image_size = -1
        self.down_rate = down_rate
        self.down = self.up = None

    @torch.no_grad()
    def cache(self, easy: EasyDataset, label: int):
        cached = self.run_or_load(self.cache_cov, easy=easy, index=label, label=label)
        self.mean[label], self.cov[label], self.image_size = cached
        self.sample[label] = self.run_or_load(self.cache_sample, index=label, label=label)

    @torch.no_grad()
    def cache_cov(self, easy: EasyDataset, label: int) -> (torch.tensor, torch.tensor, int):
        xs = []
        u_size = 0
        factory = ClassSortedFactory(easy, False, True)
        loader = DataLoader(Subset(easy.eval(), factory(label)), batch_size=1000, shuffle=False)
        for x, y in tqdm(loader):
            if self.down is None:
                u_size, d_size = x.shape[-1], x.shape[-1] // self.down_rate
                self.down = torch.nn.Upsample(size=(d_size, d_size), mode='bilinear', align_corners=False)
            indices = y.to(self._device) == label
            if indices.sum() != 0:
                xs.append(self.down(x.to(self._device)[indices]).clone().detach())
        xs = torch.cat(xs)
        xs = xs.view(len(xs), -1)
        mean = xs.mean(dim=0)
        xs = xs - mean.unsqueeze(dim=0)
        cov = xs.t() @ xs / len(xs)
        cov = cov + 1e-4 * torch.eye(len(cov)).to(self._device)
        return mean, cov, u_size

    @torch.no_grad()
    def cache_sample(self, label: int) -> torch.tensor:
        return self.new_sample(label)

    @torch.no_grad()
    def new_sample(self, label: int) -> torch.tensor:
        dist = MultivariateNormal(self.mean[label], covariance_matrix=self.cov[label])
        d_size = self.image_size // self.down_rate
        sample = dist.sample().view(1, 3, d_size, d_size)
        if self.up is None:
            self.up = torch.nn.Upsample(size=(d_size, d_size), mode='bilinear', align_corners=False)
        return self.up(sample).clone().detach()

    @torch.no_grad()
    def __call__(self, label: int, force_new: bool = False) -> torch.tensor:
        if force_new:
            return self.new_sample(label)
        return self.sample[label]
