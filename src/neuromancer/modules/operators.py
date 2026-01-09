"""Operators and neural operator wrappers."""

from __future__ import annotations

import torch
from neuralop.models import (
    FNO as _BaseFNO,
    SFNO as _BaseSFNO,
    TFNO as _BaseTFNO,
    GINO as _BaseGINO,
    UNO as _BaseUNO,
)
from torch import nn


class _StripMetadataMixin:
    def state_dict(self, *args, **kwargs):
        """Strip _metadata from state dict."""
        state = super().state_dict(*args, **kwargs)
        state.pop("_metadata", None)
        return state

    def load_state_dict(self, state, **kwargs):
        """Ignore _metadata when loading state dict."""
        # Copy to avoid mutating the caller's dict in-place
        state = dict(state)
        state.pop("_metadata", None)
        return super().load_state_dict(state, **kwargs)


class FNO(_StripMetadataMixin, _BaseFNO):
    """
    Wrapper around neuralop.models.FNO that strips ``_metadata`` from
    checkpoints for compatibility across PyTorch versions.
    """


class SFNO(_StripMetadataMixin, _BaseSFNO):
    """
    Wrapper around neuralop.models.SFNO that strips ``_metadata``.
    """


class TFNO(_StripMetadataMixin, _BaseTFNO):
    """
    Wrapper around neuralop.models.TFNO that strips ``_metadata``.
    """


class GINO(_StripMetadataMixin, _BaseGINO):
    """
    Wrapper around neuralop.models.GINO that strips ``_metadata``.
    """


class UNO(_StripMetadataMixin, _BaseUNO):
    """
    Wrapper around neuralop.models.UNO that strips ``_metadata``.
    """


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


class DeepONetCartesianProd(nn.Module):
    """
    DeepONet with Cartesian-product evaluation.

    Follows code from link below, adjusted for Neuromancer Blocks:
    https://deepxde.readthedocs.io/en/stable/modules/deepxde.nn.pytorch.html#deepxde.nn.pytorch.deeponet.DeepONetCartesianProd

    B = Batch size or NSamples
    m = number of sensors / trunk locations
    p = latent interaction dimension (for e.g MLP output)
    dim_x = input dimension for trunk net

    Accepts (leading index should be same for Neuromancer inputs,
    so we keep it batch first by PyTorch convention):
        branch_input: (B, m)
        trunk_input:  (B, m, dim_x)  # repeated grid for each sample
        output:       (B, m)  or (m, B) if return_transposed=True

    Internally converts:
        trunk_input -> (m, dim_x) - assumes all B identical (shared grid)

    Uses Neuromancer Blocks, typically MLPs:
        - branch_net: maps (B, m) -> (B, p)
        - trunk_net:  maps (m, dim_x) -> (m, p)

    Output:
    return_transposed=False → return (B,m) - PyTorch default batch first
    return_transposed=True → return (m,B) - output sensor first
    """

    def __init__(
        self,
        branch_net: nn.Module,
        trunk_net: nn.Module,
        bias: bool = True,
        return_transposed: bool = False,
    ) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.return_transposed = return_transposed

        self.bias = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, branch_input, trunk_input):
        """
        branch_input: (B, m)
        trunk_input:  (B, m, dim_x) # repeated grid for each sample

        branch_output: (B, p)
        trunk_output:  (m, p)

        fused output:  (B, m)
        """

        B, m = branch_input.shape

        # Pass through MLPs
        b = self.branch_net(branch_input)  # (B, p)
        # Trunk MLP: extract the shared grid from the first sample
        # trunk_input[0]: (m, dim_x)
        t = self.trunk_net(trunk_input[0])  # (m, p)

        # Cartesian-product fusion:
        #   For each batch function (B) and each trunk location (m):
        #       dot(b[b,:], t[m,:])
        out = torch.einsum("bp,mp->bm", b, t)
        # \\ To do: Changes for multiple input branch / trunk nets?

        # Add bias if needed
        if self.bias is not None:
            out = out + self.bias

        # Convert back to original shape if desired
        if self.return_transposed:
            return out.transpose(0, 1)  # (m,B)
        else:
            return out  # (B,m)


class DeepXDEWrapper(nn.Module):
    """
    Wrapper for DeepONet-style models in DeepXDE.

    The user must specify whether the wrapped model uses
    Cartesian-product (shared-grid) dde.nn.DeepONetCartesianProd or
    pointwise (per-sample) dde.nn.DeepONet.

    Parameters
    ----------
    model : nn.Module
        DeepONet-style model from DeepXDE that expects inputs
        as (branch_inputs, trunk_inputs).

    B = Batch size or NSamples
    m = number of sensors / trunk locations
    p = latent interaction dimension (for e.g MLP output)
    dim_x = input dimension for trunk net

    is_cartesian : bool
        If True:
            - Model is assumed to be a Cartesian-product DeepONet
            - Trunk inputs represent a shared grid
            - Accepted trunk shapes:
                * (m, dim_x)
                * (B, m, dim_x)  -> first batch element is used
        If False:
            - Model is assumed to be a basic (pointwise) DeepONet
            - Trunk inputs are per-sample
            - Required trunk shape:
                * (B, dim_x)

    branch_key, trunk_key, output_key : str
        Keys used when calling forward(batch_dict).
    """

    def __init__(
        self,
        model: nn.Module,
        is_cartesian: bool,
        branch_key: str = "branch_inputs",
        trunk_key: str = "trunk_inputs",
        output_key: str = "outputs",
    ) -> None:
        super().__init__()
        self.model = model
        self.is_cartesian = is_cartesian
        self.branch_key = branch_key
        self.trunk_key = trunk_key
        self.output_key = output_key

    def _normalize_trunk(self, trunk: torch.Tensor) -> torch.Tensor:
        """
        Normalize trunk input according to is_cartesian flag.
        """
        if self.is_cartesian:
            # Shared-grid (Cartesian-product) DeepONet
            if trunk.dim() == 3:
                # (B, m, dim_x) -> shared grid
                return trunk[0]
            elif trunk.dim() == 2:
                # (m, dim_x)
                return trunk
            else:
                raise ValueError(
                    "Cartesian DeepONet expects trunk_inputs of shape "
                    "(m, dim_x) or (B, m, dim_x)."
                )
        else:
            # Pointwise DeepONet
            if trunk.dim() != 2:
                raise ValueError(
                    "Pointwise DeepONet expects trunk_inputs of shape (B, dim_x)."
                )
            return trunk

    def forward(self, *args, **kwargs):
        # Case 1: dict batch
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            branch = batch[self.branch_key]
            trunk = batch[self.trunk_key]
            targets = batch.get(self.output_key)

        # Case 2: positional tensors
        elif len(args) >= 2:
            branch, trunk = args[:2]
            targets = kwargs.get("targets")

        else:
            raise TypeError("Expected forward(batch_dict) or forward(branch, trunk).")

        trunk = self._normalize_trunk(trunk)
        preds = self.model((branch, trunk))

        return (preds, targets) if targets is not None else preds


__all__ = [
    "FNO",
    "SFNO",
    "TFNO",
    "GINO",
    "UNO",
    "LpLoss",
    "H1Loss",
    "DeepONetCartesianProd",
    "DeepXDEWrapper",
]
