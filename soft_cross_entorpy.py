from torch.distributions import MultivariateNormal
from tqdm import tqdm 
from torch.utils import model_zoo
import pdb

from torch import nn
import torch


class DistSampler:
    def __init__(self):
        url1 = 'https://github.com/AminJun/SoftXent/releases/download/Create/mean_prob.pt'
        url2 = 'https://github.com/AminJun/SoftXent/releases/download/Create/cov.pt'
        self.mean_tensor = model_zoo.load_url(url1, map_location='cpu').cuda().double()
        self.cov_tensor = model_zoo.load_url(url2, map_location='cpu').cuda().double()
        self.samples = [MultivariateNormal(self.mean_tensor[i], covariance_matrix=self.cov_tensor).sample() for i in tqdm(range(1000))]

    def __call__(self, target: int):
        return self.samples[target]


class SoftCrossEntropy(nn.Module):
    def __init__(self, n_class: int = 1000, mode: int = 1, coef:float=1.):
        super().__init__()
        self.log_soft_max = nn.LogSoftmax(dim=1)
        self.coef = coef
        if mode != 3:
            self.load_means(pretrained=True, mode=mode)
        else:
            self.load_cov()

    def load_cov(self):
        sampler = DistSampler()
        self.label = torch.stack([sampler(i) for i in range(1000)])

    def load_means(self, checkpoint: str = None, pretrained: bool = True, mode: int = 1):
        if pretrained:
            url = ['https://github.com/AminJun/SoftXent/releases/download/Create/mean_prob.pt',
                   'https://github.com/AminJun/SoftXent/releases/download/Create/e1.pt',
                   'https://github.com/AminJun/SoftXent/releases/download/Create/t2.pt', ][mode]
            self.label = model_zoo.load_url(url, map_location='cpu').cuda()
        elif checkpoint is not None:
            self.label = torch.load(checkpoint).cuda()

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        if self.label is not None:
            target = self.label[target]
        return - torch.sum(target * self.log_soft_max(prediction), dim=1).mean() * self.coef


def main():
    pred = torch.load('checkpoints/xs.pt').cuda()
    labels = torch.load('checkpoints/ys.pt').cuda()
    loss2 = nn.CrossEntropyLoss()
    l2 = loss2(pred, labels)
    for i in range(4):
        loss1 = SoftCrossEntropy(mode=i).cuda()
        l1 = loss1(pred, labels)
        print(l1.item(), l2.item())


if __name__ == '__main__':
    main()
