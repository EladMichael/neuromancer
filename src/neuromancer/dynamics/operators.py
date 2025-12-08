from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
from torch import nn

if TYPE_CHECKING:
    from neuromancer.modules.blocks import Block

TDeepONet = TypeVar("TDeepONet", bound="DeepONet")


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

        # # Convert to batch-first for PyTorch MLPs
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


class DeepXDEDataWrapper(nn.Module):
    """
    Adapter that makes DeepXDE models compatible with NeuroMANCER Nodes.

    DeepXDE expects a single tuple/list input (branch_inputs, trunk_inputs)

    B = Batch size or NSamples
    m = number of sensors / trunk locations
    p = latent interaction dimension (for e.g MLP output)
    dim_x = input dimension for trunk net

    Step1: Convert trunk input from (B, m, dim_x) to (m, dim_x)
    assuming all B identical (shared grid). 2D case is for error handling.
    Step2: Create the Tuple input (branch_inputs, trunk_inputs) where
           branch_inputs = (batch, m)
           trunk_inputs = (m, dim_x)
    Step3: Call the Neuromancer model with the tuple input
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, branch_inputs: torch.Tensor, trunk_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        :param branch_inputs: Tensor shaped (batch, m)
        :param trunk_inputs: Tensor shaped (m, dim_x) or (batch, m, dim_x)
                             with identical grids
        """
        if trunk_inputs.dim() == 3:
            trunk_inputs = trunk_inputs[0]
        return self.model((branch_inputs, trunk_inputs))


__all__ = ["DeepONetCartesianProd", "DeepXDEDataWrapper"]
