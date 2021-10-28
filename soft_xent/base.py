import torch
from torch import nn


class SoftLabelData:
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, n_classes: int, mode: int = -1, ):
        self.mode = mode
        self.n = n_classes

    def __call__(self, mode: int = -1):
        mode = max(mode, self.mode)
        return self.get_labels(mode)

    def get_labels(self, mode: int) -> torch.tensor:
        modes = [self.get_1_eval, self.get_1_train, self.get_mean, self.get_dist, self.one_hot]
        return modes[mode]()

    def get_1_eval(self) -> torch.tensor:
        raise NotImplementedError

    def get_1_train(self) -> torch.tensor:
        raise NotImplementedError

    def get_mean(self) -> torch.tensor:
        raise NotImplementedError

    def get_dist(self) -> torch.tensor:
        raise NotImplementedError

    def one_hot(self) -> torch.tensor:
        return torch.eye(self.n).to(self._device)


class SoftCrossEntropy(nn.Module):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, label: torch.tensor, coef: float = 1.):
        super().__init__()
        self.log_soft_max = nn.LogSoftmax(dim=1)
        self.coef = coef
        self.label = label

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        target = self.label[target]
        return - torch.sum(target * self.log_soft_max(prediction), dim=1).mean() * self.coef
