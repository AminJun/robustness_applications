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
        modes = [self.get_one_e, self.get_one_t, self.get_mean_t, self.get_mean_e,
                 self.get_dist_e, self.get_dist_t, self.one_hot]
        return modes[mode]()

    def get_one_e(self) -> torch.tensor:
        raise NotImplementedError

    def get_one_t(self) -> torch.tensor:
        raise NotImplementedError

    def get_mean_t(self) -> torch.tensor:
        raise NotImplementedError

    def get_mean_e(self) -> torch.tensor:
        raise NotImplementedError

    def get_dist_t(self) -> torch.tensor:
        raise NotImplementedError

    def get_dist_e(self) -> torch.tensor:
        raise NotImplementedError

    def one_hot(self) -> torch.tensor:
        return torch.eye(self.n).to(self._device)


class SoftCrossEntropy(nn.Module):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, label: torch.tensor, coef: float = 1., reduction: str = 'mean'):
        super().__init__()
        self.log_soft_max = nn.LogSoftmax(dim=1)
        self.coef = coef
        self.label = label
        self.reduction = reduction

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        target = self.label[target]
        loss = -torch.sum(target * self.log_soft_max(prediction)) * self.coef
        if self.reduction is 'mean':
            loss = loss.mean()
        return loss
