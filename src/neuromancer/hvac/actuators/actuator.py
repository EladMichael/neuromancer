"""
actuator.py

General Actuator class for HVAC components with Zone Vectorization Support.
Supports different levels of fidelity for different use cases while maintaining
a consistent interface.

ZONE VECTORIZATION SUPPORT:
- Supports 1 to n_zones with zone-specific or shared parameters
- Input tensors expected as [batch_size, n_zones]
- Output tensors produced as [batch_size, n_zones]
- Zone-specific parameters automatically expanded from scalars if needed
"""

import torch
from typing import Literal, Union, List
from .._runtime import beartype


class Actuator(torch.nn.Module):
    """
    General actuator class with pluggable dynamics models and zone vectorization.

    Supports two modeling approaches:
    - "instantaneous": No lag, output = setpoint immediately
    - "analytic": Exact closed-form solution to the first-order lag,
                  x(t+dt) = setpoint + (x - setpoint) * exp(-dt/tau).
                  Exact, fast, and differentiable in tau (suitable for learning).

    Zone Vectorization:
        - Handles multiple zones simultaneously with zone-specific parameters
        - All tensor inputs/outputs have shape [batch_size, n_zones]
        - Parameters can be scalars (shared) or vectors (zone-specific)
        - Automatic broadcasting ensures compatibility between parameters and inputs

    Units:
        position: [0-1] Normalized actuator position
        setpoint: [0-1] Normalized actuator setpoint
        tau: [s] Time constant (per zone)
        t: [s] Time

    Tensor Shapes:
        Input tensors: [batch_size, n_zones]
        Output tensors: [batch_size, n_zones]
        Parameters: [n_zones] for zone-specific, scalar for shared
    """
# TODO: Offset in addition to time lag
    def __init__(
            self,
            tau: Union[float, List[float], torch.Tensor] = 15.0,
            # [s] Time constant per zone
            # Can be:
            #   - Scalar: Same time constant for all zones
            #   - List[n_zones]: Zone-specific time constants
            #   - Tensor[n_zones]: Zone-specific time constants
            # Typical: 5-15 s for electric actuators, 10-30 s for pneumatic
            model: Literal["instantaneous", "analytic"] = "analytic",
            # Dynamics model type
            # "instantaneous": No lag (immediate response)
            # "analytic": Exact first-order lag solution

            name: str = "actuator",
            # Actuator name for identification and debugging
    ):
        """
        Initialize actuator with specified dynamics model.

        Zone Vectorization:
            Parameters can be provided as scalars (shared across zones) or as
            lists/tensors (zone-specific values). The BuildingComponent base class
            handles automatic expansion of scalar parameters to zone vectors.

        Args:
            tau: Time constant [s] - scalar (shared) or vector (zone-specific)
            model: Dynamics model type
            name: Actuator name for identification
        """
        super().__init__()
        self.model = model
        self.name = name
        # Normalize tau to a tensor so the analytic lag works for bare actuators too.
        # When constructed via a BuildingComponent, tau already arrives as a tensor
        # (or learnable Parameter); as_tensor preserves it (and its autograd).
        self.tau = tau if isinstance(tau, torch.Tensor) else torch.as_tensor(tau, dtype=torch.float32)
        # Validate model type
        valid_models = ["instantaneous", "analytic"]
        if model not in valid_models:
            raise ValueError(f"model must be one of {valid_models}, got {model}")

    def forward(
            self,
            t: float = 0.,  # [s] Current time
            setpoint: torch.Tensor = None,  # [0-1] Desired actuator position, shape [batch_size, n_zones]
            position: torch.Tensor = None,  # [0-1] Current position, shape [batch_size, n_zones]
            dt: float = None,  # [s] Time step (for analytic/smooth models)
    ) -> torch.Tensor:
        """
        Compute actuator response with zone vectorization support.

        Args:
            t (float): Current simulation time [s]
            setpoint (Tensor): Desired actuator position [0-1], shape [batch_size, n_zones]
            position (Tensor): Current actuator position [0-1], shape [batch_size, n_zones]
                              Required for non-instantaneous models.
            dt (float): Time step [s]. Used for analytic/smooth models.

        Returns:
            Tensor: New actuator position [0-1], shape [batch_size, n_zones]
        """
        assert setpoint is not None, "Cannot call actuator without setpoint"

        if self.model == "instantaneous":
            return self._forward_instantaneous(setpoint)
        elif self.model == "analytic":
            return self._forward_analytic(setpoint, position, dt)
        else:
            raise ValueError(f"Unknown model type: {self.model}")

    def _forward_instantaneous(self, setpoint: torch.Tensor) -> torch.Tensor:
        """
        Instantaneous response - no lag.

        Args:
            setpoint: Desired position [0-1], shape [batch_size, n_zones]

        Returns:
            Tensor: New position [0-1], shape [batch_size, n_zones]
        """
        return setpoint

    def _forward_analytic(self, setpoint: torch.Tensor, position: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Analytic solution to first-order lag (exact solution).

        Uses the exact mathematical solution: x(t+dt) = setpoint + (x₀ - setpoint) * exp(-dt/tau)

        Args:
            position: Current position [0-1], shape [batch_size, n_zones]
            setpoint: Desired position [0-1], shape [batch_size, n_zones]
            dt: Time step [s]

        Returns:
            Tensor: New position [0-1], shape [batch_size, n_zones]
        """
        assert position is not None and dt is not None, "Analytic model requires position and dt"

        # Analytic solution: x(t+dt) = setpoint + (x_current - setpoint) * exp(-dt/tau)
        # Broadcasting: scalar / [n_zones] -> [n_zones]
        decay_factor = torch.exp(-dt / self.tau)
        # Broadcasting: [batch_size, n_zones] + ([batch_size, n_zones] - [batch_size, n_zones]) * [n_zones]
        position_new = setpoint + (position - setpoint) * decay_factor

        return position_new
