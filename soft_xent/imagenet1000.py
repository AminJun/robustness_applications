from torch.distributions import MultivariateNormal
from tqdm import tqdm
from torch.utils import model_zoo
from .base import SoftLabelData, SoftCrossEntropy

import torch


class ImageNet1000NatData(SoftLabelData):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, mode: int = -1):
        super().__init__(n_classes=1000, mode=mode)

    def get_one_e(self) -> torch.tensor:
        url = 'https://github.com/AminJun/SoftXent/releases/download/Create/e1.pt'
        return model_zoo.load_url(url, map_location='cpu').to(self._device)

    def get_one_t(self) -> torch.tensor:
        url = 'https://github.com/AminJun/SoftXent/releases/download/Create/t2.pt'
        return model_zoo.load_url(url, map_location='cpu').to(self._device)

    def get_mean_e(self) -> torch.tensor:
        url = 'https://github.com/AminJun/SoftXent/releases/download/Create/mean_prob.pt'
        return model_zoo.load_url(url, map_location='cpu').to(self._device)

    def get_dist_e(self) -> torch.tensor:
        url1 = 'https://github.com/AminJun/SoftXent/releases/download/Create/mean_prob.pt'
        url2 = 'https://github.com/AminJun/SoftXent/releases/download/Create/cov.pt'
        mean = model_zoo.load_url(url1, map_location='cpu').to(self._device).double()
        cov = model_zoo.load_url(url2, map_location='cpu').to(self._device).double()
        out = torch.cat([MultivariateNormal(mean[i], covariance_matrix=cov).sample() for i in tqdm(range(self.n))])
        return out.to(self._device)

    def get_mean_t(self) -> torch.tensor:
        raise NotImplementedError

    def get_dist_t(self) -> torch.tensor:
        raise NotImplementedError


class SoftXentIN1000(SoftCrossEntropy):
    def __init__(self, mode: int = 4, coef: float = 1.0):
        data = ImageNet1000NatData(mode=mode)
        super().__init__(label=data(), coef=coef)
