import torch
from torch.distributions import MultivariateNormal
from torch.utils import model_zoo


class ImageNetMultiVariate:
    _url = 'https://github.com/AminJun/InversionInitNoise/releases/download/main/{mean_vs_cov}{no}.pt'
    _n_classes = 9
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.mean = [model_zoo.load_url(self._url.format(mean_vs_cov='mean', no=i), map_location='cpu').to(self._device)
                     for i in range(self._n_classes)]
        self.covs = [model_zoo.load_url(self._url.format(mean_vs_cov='cov', no=i), map_location='cpu').to(self._device)
                     for i in range(self._n_classes)]
        self.dist = [MultivariateNormal(mean, covariance_matrix=cov) for mean, cov in zip(self.mean, self.covs)]

    def __getitem__(self, item: int) -> MultivariateNormal:
        return self.dist[item]

    def __len__(self):
        return len(self.dist)
