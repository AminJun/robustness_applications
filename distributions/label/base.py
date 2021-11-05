import torch
from torch import nn


class SoftLabelData:
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, n_classes: int, mode: int = -1, ):
        self.mode = mode
        self.n = n_classes

    def __call__(self, mode: int = -1, force_new: bool = False):
        mode = max(mode, self.mode)
        return self.get_labels(mode, force_new)

    def get_labels(self, mode: int, force_new: bool = False) -> torch.tensor:
        modes = [self.get_one_e, self.get_one_t, self.get_mean_t, self.get_mean_e,
                 self.get_dist_e, self.get_dist_t, self.one_hot, self.get_logit_t, self.get_logit_e]
        return modes[mode](force_new=force_new)

    def get_one_e(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_one_t(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_mean_t(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_mean_e(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_dist_t(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_dist_e(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_logit_t(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def get_logit_e(self, force_new: bool = False) -> torch.tensor:
        raise NotImplementedError

    def one_hot(self, *args, **kwargs) -> torch.tensor:
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
        loss = -torch.sum(target * self.log_soft_max(prediction), dim=-1) * self.coef
        if self.reduction is 'mean':
            loss = loss.mean()
        return loss
