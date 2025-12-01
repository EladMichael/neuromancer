"""
Operators and neural operator wrappers.
"""

from neuralop.models import FNO as _BaseFNO
import torch


class FNO(_BaseFNO):
    """
    Wrapper around neuralop.models.FNO that strips ``_metadata`` from
    checkpoints for compatibility across PyTorch versions
    \\To Do: Check whether this is needed with all available NOs in the neuralop.models
    """

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.pop("_metadata", None)
        return state

    def load_state_dict(self, state, **kwargs):
        # Copy to avoid mutating the caller's dict in-place
        state = dict(state)
        state.pop("_metadata", None)
        return super().load_state_dict(state, **kwargs)


class LpLoss(object):
    """
    Lp loss class, for computing relative or absolute Lp losses
    over spatial dimensions d.
    Args:
        d (int): spatial dimensions
        p (int): Lp norm type
        size_average (bool): if true, average over batch
        reduction (bool): if true, reduce over batch
    Usage:
        loss = LpLoss(d=2, p=2)
        abs_loss = loss.abs(x, y)
        rel_loss = loss.rel(x, y)
        Default is relative Lp loss: loss(x, y)
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        # Flatten (batch, spatial...) to (batch, -1)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class H1Loss(object):
    """
    H1 loss class for computing H1 losses over spatial dimensions d.
    Args:
        d (int): spatial dimensions
        beta (float): weight for gradient term
    Usage:
        loss = H1Loss(d=2, beta=1.0)
        h1_loss = loss(x, y)
    1. Standard L2 (value) error
    2. Finite forward differences for gradients
    3. Relative gradient L2 error
    4. Combine
    --------------------------------------------------------------------------
    1. Standard L2 (value) error
        L2_loss = ||x - y||_2
    2. Finite forward differences for gradients
        dx_x = x[..., 1:, :] - x[..., :-1, :]
        dx_y = y[..., 1:, :] - y[..., :-1, :]
    """

    def __init__(self, d=2, beta=1.0):
        self.d = d
        self.beta = beta
        self.l2 = LpLoss(d=d, p=2)

    def __call__(self, x, y):
        # 1. Standard L2 (value) error
        l2_loss = self.l2(x, y)

        # 2. Compute Gradients (Central Difference)
        # Assumes shape [Batch, ..., X, Y]
        # We compute dy/dx and dy/dy for both Pred (x) and True (y)

        # 2. Finite forward differences for gradients
        dx_x = x[..., 1:, :] - x[..., :-1, :]
        dx_y = y[..., 1:, :] - y[..., :-1, :]

        dy_x = x[..., :, 1:] - x[..., :, :-1]
        dy_y = y[..., :, 1:] - y[..., :, :-1]

        # 3. Relative gradient L2 error
        term_x = torch.norm(dx_x - dx_y, p=2) / torch.norm(dx_y, p=2)
        term_y = torch.norm(dy_x - dy_y, p=2) / torch.norm(dy_y, p=2)

        # 4. Combine
        return l2_loss + self.beta * (term_x + term_y)


__all__ = ["FNO", "LpLoss", "H1Loss"]
