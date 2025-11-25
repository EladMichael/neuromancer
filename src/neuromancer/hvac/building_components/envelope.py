"""
envelope.py

This module contains differentiable, continuous-time models for the thermal envelope of buildings,
using resistor-capacitor (RC) network representations suitable for simulation, system identification,
and control in HVAC and building energy applications.

ZONE VECTORIZATION SUPPORT:
- Supports 1 to n_zones with zone-specific or shared parameters
- Input tensors expected as [batch_size, n_zones]
- Output tensors produced as [batch_size, n_zones]
- Zone-specific parameters automatically expanded from scalars if needed
"""

import torch
import numpy as np
from torchdiffeq import odeint
import neuromancer.hvac.simclock as simclock
from .base import BuildingComponent
import os
if os.environ.get("RUNTIME_TYPING", 1):
    from beartype import beartype 
else:
    # passthrough for no type checking
    def beartype(fn): return fn


class Envelope(BuildingComponent):
    """
    Physics-based differentiable model of a multi-zone building thermal envelope with integrated dynamics.

    This component models building thermal envelope using resistor-capacitor (RC) networks with internal
    ODE integration for temperature dynamics. Each zone has thermal capacitance (thermal mass) and thermal
    resistance (envelope insulation). Zones can exchange heat with each other and the ambient environment.

    ZONE VECTORIZATION SUPPORT:
    - Handles multiple zones simultaneously with zone-specific parameters
    - All tensor inputs/outputs have shape [batch_size, n_zones]
    - Parameters can be scalars (shared) or vectors (zone-specific)
    - Automatic broadcasting ensures compatibility between parameters and inputs

    States (maintained by component):
        - T_zones: Zone air temperatures [K] (computed state with internal dynamics)

    External Inputs (shape [batch_size, n_zones] unless noted):
        - T_outdoor: Outdoor air temperature [K] (can be [batch_size, 1] for broadcast)
        - Q_solar: Solar heat gain per zone [W]
        - Q_internal: Internal heat gains from occupancy and equipment [W]
        - Q_hvac: HVAC heat input/output per zone [W] (negative=cooling, positive=heating)

    Zone-Specific Parameters (expandable to [n_zones] vectors):
        - R_env: Envelope thermal resistance per zone [K/W]
        - C_env: Thermal capacitance per zone [J/K]

    Shared Parameters (scalars):
        - R_internal: Inter-zone thermal resistance [K/W]
        - adjacency_threshold: Threshold for discretizing learned topology [0-1]
        - ode_method: ODE integration method (e.g., 'dopri5', 'rk4')
        - ode_rtol: Relative tolerance for ODE solver
        - ode_atol: Absolute tolerance for ODE solver

    Primary Outputs (shape [batch_size, n_zones]):
        - T_zones: Zone air temperatures after dynamics integration [K]

    Diagnostic Outputs (available via .diagnostics property):
        - dT_zones_dt: Zone temperature derivatives [K/s]
        - Q_env_exchange: Heat exchange with ambient environment [W]
        - total_heat_input: Total heat input from solar, internal, and HVAC [W]

    Physical Modeling:
        - RC network thermal dynamics with configurable zone connectivity
        - Heat exchange with ambient environment through envelope resistance
        - Inter-zone heat transfer through internal resistance matrix
        - Optional learnable adjacency matrix for zone connectivity topology
        - Internal ODE integration handles temperature dynamics automatically

    ODE Integration:
        - Uses torchdiffeq for robust numerical integration
        - Configurable solver method and tolerances
        - Integration performed internally during each forward() call
        - Time step dt provided as input to forward() method

    Units:
        - Temperature: Kelvin [K]
        - Time: Seconds [s]
        - Thermal resistance: Kelvin per Watt [K/W]
        - Thermal capacitance: Joule per Kelvin [J/K]
        - Heat gains/losses: Watt [W]
        - dT/dt: Kelvin per second [K/s]

    Tensor Shapes:
        Input tensors: [batch_size, n_zones]
        Output tensors: [batch_size, n_zones]
        Parameters: [n_zones] for zone-specific, scalar for shared
    """
    # Variable ranges for validation and initialization
    _state_ranges = {
        "T_zones": (283.15, 323.15),  # [K] Zone air temperature (computed state)
    }
    _external_ranges = {
        "T_outdoor": (253.15, 318.15),  # [K] Outdoor air temperature
        "Q_solar": (0.0, 2000.0),  # [W] Solar heat gain per zone
        "Q_internal": (0.0, 2000.0),  # [W] Internal heat gain per zone
        "Q_hvac": (-5000.0, 5000.0),  # [W] HVAC heat/cool addition per zone
    }
    _zone_param_ranges = {
        # Zone-specific parameters (expanded to [n_zones] vectors)
        "R_env": (0.05, 2.0),  # [K/W] Envelope resistance per zone
        "C_env": (1e5, 5e7),   # [J/K] Envelope capacitance per zone
    }
    _param_ranges = {
        # Shared parameters (scalars)
        "R_internal": (0.01, 1.0),      # [K/W] Inter-zone resistance
        "adjacency_threshold": (0.0, 1.0),  # [0-1] Threshold for adjacency
        # ODE solver parameters
        "ode_rtol": (1e-8, 1e-3),  # Relative tolerance
        "ode_atol": (1e-10, 1e-5), # Absolute tolerance
    }

    @beartype
    def __init__(
            self,
            n_zones: int = 1,
            # Zone-specific parameters (can be scalar or list[n_zones])
            R_env: float | list = 0.1,     # [K/W] Envelope resistance per zone
            C_env: float | list = 1e6,     # [J/K] Thermal capacitance per zone
            # Shared parameters
            R_internal: float | list | np.ndarray | torch.Tensor = 0.02,  # [K/W] Inter-zone resistances
            adjacency_threshold: float = 0.5,  # [0-1] Adjacency threshold
            # ODE solver parameters
            ode_method: str = 'dopri5',  # ODE integration method
            ode_rtol: float = 1e-5,      # Relative tolerance
            ode_atol: float = 1e-7,      # Absolute tolerance
            # Special adjacency matrix parameter
            adjacency: list | np.ndarray | torch.Tensor = None,  # [n_zones, n_zones] Zone connectivity
            # Standard parameters
            learnable: set = None,
            device: torch.device = None,
            dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Envelope component with zone vectorization support and integrated ODE dynamics.

        ZONE PARAMETER HANDLING:
        - Scalar parameters are automatically expanded to all zones by the base class
        - List parameters must have length n_zones for zone-specific values
        - Sub-components receive properly shaped parameter tensors

        Args:
            n_zones (int): Number of building zones.
            R_env (float or list): Envelope thermal resistance [K/W].
                Controls heat exchange rate between zones and outdoor environment.
            C_env (float or list): Thermal capacitance [J/K].
                Controls thermal mass and temperature response rate of each zone.
            R_internal (float or Iterable): Inter-zone thermal resistances [K/W].
                Controls heat transfer rate between adjacent zones.
            adjacency_threshold (float): Threshold for discretizing learned adjacency matrix [0-1].
                Used during evaluation to convert continuous connectivity to discrete.
            ode_method (str): ODE integration method for temperature dynamics.
                Options: 'dopri5' (adaptive), 'rk4' (fixed), 'euler' (simple).
            ode_rtol (float): Relative tolerance for ODE solver accuracy.
            ode_atol (float): Absolute tolerance for ODE solver accuracy.
            adjacency (Tensor, optional): Initial zone connectivity matrix [n_zones, n_zones].
                If None, defaults to all zones connected (except self-connections).
            learnable (dict): Parameters to make learnable for optimization/learning applications.
            device (torch.device): Device for tensor computations.
            dtype (torch.dtype): Tensor data type for computation.
        """

        # Handle adjacency matrix setup before calling super().__init__
        # Create connection indices for upper triangular matrix (excluding diagonal)

        connection_indices = torch.triu_indices(n_zones, n_zones, offset=1)

        # Handle adjacency matrix parameter setup
        learnable = set(learnable) if learnable else set()

        if adjacency is None:
            # Default: all zones connected except self-connections
            adjacency = torch.ones((n_zones, n_zones), dtype=dtype, device=device)
            adjacency.fill_diagonal_(0)
        else:
            adjacency = torch.tensor(adjacency)

        # Store adjacency representation based on whether it's learnable
        if 'adjacency' in learnable:
            # Learnable adjacency: store as logits for upper triangular elements
            adj_triu = adjacency[connection_indices[0], connection_indices[1]]
            adj_logits = torch.logit(adj_triu.clamp(1e-4, 1 - 1e-4))
            # Replace 'adjacency' with internal representation in learnable set
            learnable.remove('adjacency')
            learnable.add('adj_logits')
        else:
            # Fixed adjacency: store full matrix
            adj_logits = torch.logit(adjacency.clamp(1e-4, 1 - 1e-4))

        if isinstance(R_internal, float):
            R_internal = torch.ones(n_zones, n_zones) * R_internal
        else:
            R_internal = torch.tensor(R_internal)
            assert R_internal.shape == (n_zones, n_zones), f"Internal Resista\
            nce [Shape {R_internal.shape}] must have shape (n_zones, n_zones)"

        # Handle R_internal similar to adjacency if it's learnable
        if 'R_internal' in learnable:
            # Learnable R_internal: store as log values to ensure positivity
            R_internal_logits = torch.log(torch.tensor(R_internal, dtype=dtype, device=device))
            learnable.remove('R_internal')
            learnable.add('R_internal_logits')
        else:
            # Store fixed resistances
            assert torch.all(R_internal >= 0), "Internal resistances must be non-negative"
            R_internal_logits = torch.log(R_internal.clamp(1e-8, torch.inf))

        super().__init__(params=locals(), learnable=learnable, device=device, dtype=dtype)

        # Store connection indices as buffer
        self.register_buffer('connection_indices', connection_indices)

    @beartype
    def _reconstruct_symmetric_matrix(self, connection_values: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a symmetric matrix from upper-triangular connection values.

        Args:
            connection_values (Tensor): Values for upper-triangular connections,
                                       shape (n_connections,)

        Returns:
            Tensor: Symmetric matrix (n_zones, n_zones) with zeros on diagonal
        """
        # Create zero matrix with same device/dtype as connection values
        matrix = torch.zeros((self.n_zones, self.n_zones),
                             device=connection_values.device,
                             dtype=connection_values.dtype)

        # Fill upper triangular from connection parameters
        matrix[self.connection_indices[0], self.connection_indices[1]] = connection_values

        # Make symmetric: matrix[i,j] = matrix[j,i]
        return matrix + matrix.T

    @property
    def R_internal_matrix(self) -> torch.Tensor:
        """
        Get resistance matrix

        Returns:
            Tensor: Full resistance matrix (n_zones, n_zones) with zeros on diagonal.
                   All resistance values are guaranteed positive.
        """
        
        # Exponential ensures positive resistances
        R_values = torch.exp(self.R_internal_logits)
        # Create full matrix with this resistance value
        R_matrix = torch.full((self.n_zones, self.n_zones), R_values.item(),
                              device=self.device, dtype=self.dtype)
        R_matrix.fill_diagonal_(0)  # No self-connections
        return R_matrix

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """
        Get adjacency matrix with optimized path for fixed vs learnable cases.

        Returns:
            Tensor: Adjacency matrix (n_zones, n_zones) with zeros on diagonal.
                   During training: continuous values in [0,1] for gradients
                   During evaluation: discrete values {0,1} after thresholding
        """
        if 'adj_logits' in self._parameters:
            # Learnable: reconstruct from upper triangular logits
            connections = torch.sigmoid(self.adj_logits)
            if not self.training:  # Discretize during evaluation
                connections = (connections > self.adjacency_threshold).float()
            return self._reconstruct_symmetric_matrix(connections)
        else:
            # Fixed: return stored matrix directly
            return torch.sigmoid(self.adj_logits)

    @beartype
    def ode_rhs(
            self,
            t: float,  # [s] Current simulation time
            T_zones: torch.Tensor,  # [K] Zone temperatures, shape [batch_size, n_zones]
    ) -> torch.Tensor:
        """
        Calculate temperature derivatives for ODE integration (internal physics computation).

        Computes dT/dt for each zone based on RC network thermal dynamics including:
        - Heat exchange with ambient environment through envelope resistance
        - Inter-zone heat transfer through internal resistance matrix
        - Heat inputs from solar, internal, and HVAC sources

        Args:
            t (float): Current simulation time [s].
            T_zones (Tensor): Zone temperatures [K], shape [batch_size, n_zones].

        Returns:
            Tensor: Temperature derivatives [K/s], shape [batch_size, n_zones].
        """
        # Use stored inputs from current forward() call
        T_outdoor = self.current_inputs['T_outdoor']
        Q_solar = self.current_inputs['Q_solar']
        Q_internal = self.current_inputs['Q_internal']
        Q_hvac = self.current_inputs['Q_hvac']

        # # Ensure T_outdoor broadcasts properly - expand if needed
        # if T_outdoor.shape[-1] == 1:
        #     T_outdoor = T_outdoor.clone.expand_as(T_zones)

        # Envelope heat exchange with ambient
        # Q_env_exchange: [W] = ([K] - [K]) / [K/W] = [W]
        # Broadcasting: [batch_size, n_zones] / [n_zones] -> [batch_size, n_zones]
        Q_env_exchange = (T_outdoor - T_zones) / self.R_env

        # Inter-zone heat exchange calculation
        # T_i: [batch_size, n_zones, 1], T_j: [batch_size, 1, n_zones]
        T_i = T_zones.unsqueeze(-1)  # [batch_size, n_zones, 1]
        T_j = T_zones.unsqueeze(-2)  # [batch_size, 1, n_zones]
        delta_T = T_j - T_i  # [batch_size, n_zones, n_zones]

        # Heat flow: Q_ij = (T_j - T_i) / R_ij * adjacency_ij
        # Both R_internal and adjacency have zeros on diagonal (no self-connections)
        R_matrix = self.R_internal_matrix  # [n_zones, n_zones]
        adj_matrix = self.adjacency_matrix  # [n_zones, n_zones]

        # Broadcasting: [batch_size, n_zones, n_zones] / [n_zones, n_zones] * [n_zones, n_zones]
        flow_ij = (delta_T / (R_matrix + 1e-8)) * adj_matrix  # Small epsilon for numerical stability

        # Sum heat flows into each zone (sum over j, the last dimension)
        Q_inter_zone = flow_ij.sum(-1)  # [batch_size, n_zones]

        # Zone temperature derivative: dT/dt = Q_total / C
        # Broadcasting: [batch_size, n_zones] / [n_zones] -> [batch_size, n_zones]
        dT_zones_dt = (Q_env_exchange + Q_solar + Q_internal + Q_hvac + Q_inter_zone) / self.C_env

        return dT_zones_dt

    @beartype
    def forward(
            self, *,
            t: float,  # [s] Current simulation time
            T_zones: torch.Tensor, # [K] Zone temperatures, shape[batch_size, n_zones]
            T_outdoor: torch.Tensor,  # [K] Outdoor temperature, [batch_size, 1]
            Q_solar: torch.Tensor,  # [W] Solar gains, shape [batch_size, n_zones]
            Q_internal: torch.Tensor,  # [W] Internal gains, shape [batch_size, n_zones]
            Q_hvac: torch.Tensor,  # [W] HVAC input, shape [batch_size, n_zones]
            dt: float = 1.0,  # [s] Time step for integration
    ) -> dict:
        """
        Calculate zone temperatures after thermal dynamics integration for one simulation time step.

        Performs internal ODE integration to advance zone temperatures from current state to
        new state after time step dt. Handles complete RC network thermal dynamics including
        envelope heat exchange, inter-zone heat transfer, and all heat inputs.

        CONTROL SEQUENCE:
        1. Store current heat inputs for use during ODE integration
        2. Integrate zone temperature dynamics over time step dt using specified ODE solver
        3. Update component state with new zone temperatures
        4. Calculate diagnostics for monitoring and analysis

        TENSOR SHAPES:
        - All zone-specific inputs: [batch_size, n_zones]
        - Ambient temperature: [batch_size, n_zones] or [batch_size, 1] (broadcast)
        - All outputs: [batch_size, n_zones]

        Args:
            t (float): Current simulation time [s].
            T_outdoor (Tensor): Outdoor air temperatures [K], shape [batch_size, n_zones] or [batch_size, 1].
            Q_solar (Tensor): Solar heat gains [W], shape [batch_size, n_zones].
            Q_internal (Tensor): Internal heat gains [W], shape [batch_size, n_zones].
            Q_hvac (Tensor): HVAC heat input/output [W], shape [batch_size, n_zones].
            dt (float): Time step [s] for ODE integration.

        Returns:
            dict: Updated zone temperatures and diagnostics, all tensors shape [batch_size, n_zones].
                T_zones: Zone temperatures after dynamics integration [K]
        """
        # Store current inputs for use in ode_rhs
        self.current_inputs = {
            'T_outdoor': T_outdoor,
            'Q_solar': Q_solar,
            'Q_internal': Q_internal,
            'Q_hvac': Q_hvac,
        }

        # Integrate over the time step
        t_span = torch.tensor([t, t + dt], device=self.device, dtype=self.dtype)

        # ODE integration
        solution = odeint(
            func=lambda time, states: self.ode_rhs(float(time), states),
            y0=T_zones,
            t=t_span,
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )

        # Extract final temperatures
        return {
            "T_zones": solution[-1],
        }

    @beartype
    def adjacency_regularization(self, reg_type: str = 'l1') -> torch.Tensor:
        """
        Regularization loss for the adjacency connections to encourage sparsity.

        Args:
            reg_type (str): 'l1' (default) or 'l2'.

        Returns:
            Tensor: Scalar regularization loss.
        """
        if 'adj_logits' not in self._parameters:
            return torch.tensor(0., device=self.device)

        # Regularize the adjacency connection probabilities (not logits)
        adj_connections = torch.sigmoid(self.adj_logits)

        if reg_type == 'l1':
            return adj_connections.sum()
        elif reg_type == 'l2':
            return (adj_connections ** 2).sum()
        else:
            raise ValueError("Unknown reg_type. Use 'l1' or 'l2'.")

    @beartype
    def initial_state_functions(self, mode: str = "steady_state"):
        """
        Return functions for sampling intelligent initial states for zone temperatures using context.

        Args:
            mode: Initialization strategy
                - "realistic": Realistic zone temperatures with small variations
                - "steady_state": Ideal steady-state temperatures near setpoints
        """
        return {
            "T_zones": lambda bs: self._sample_T_zones(bs, mode),
        }

    @beartype
    def _sample_T_zones(self, batch_size: int, mode: str):
        """Sample initial zone temperatures based on context."""
        # Use context setpoint if available, otherwise use default comfortable temperature
        T_setpoint = self.context["T_setpoint_base"]
        base_temp = torch.full((batch_size, self.n_zones), T_setpoint,
                               device=self.device, dtype=self.dtype)

        if mode == "steady_state":
            return base_temp
        elif mode == "realistic":
            # Add small random variations around setpoint (±1K)
            noise = torch.normal(0.0, 0.5, (batch_size, self.n_zones),
                                 device=self.device, dtype=self.dtype)
            return torch.clamp(base_temp + noise, T_setpoint - 2.0, T_setpoint + 2.0)

    @property
    def input_functions(self):
        """
        Context-aware input functions for Envelope component.
        Returns functions that generate tensors of shape [batch_size, n_zones].

        Returns:
            dict: Mapping from input variable names to callables (t, batch_size) -> torch.Tensor[batch_size, n_zones].
        """
        if not hasattr(self, '_input_functions'):

            # Get context values, error out if not available
            req_keys = ['T_outdoor', 'occupancy_state', 'system_mode']
            assert all(key in self.context for key in req_keys), "Context does not contain required keys!"

            T_outdoor_base = self.context.get("T_outdoor")
            occupancy_state = self.context.get("occupancy_state")
            system_mode = self.context.get("system_mode")

            def Q_solar_fn(t, batch_size=1):
                """Context-aware solar heat gain with realistic daily and seasonal patterns."""
                h_of_day = simclock.hour_of_day(t)
                d_of_year = simclock.day_of_year(t)
                # Daily solar pattern (zero at night, peak at solar noon)
                if 6 <= h_of_day <= 18:  # Daylight hours
                    daily_solar = torch.sin(torch.tensor(torch.pi * (h_of_day - 6) / 12))  # Peak at noon
                else:
                    daily_solar = torch.tensor(0.0)

                # Seasonal variation (stronger in summer, weaker in winter)
                seasonal_factor = 1.0 + 0.5 * torch.sin(torch.tensor(2 * torch.pi * (d_of_year - 80) / 365))

                # Weather factor from context affects solar intensity
                weather_factor = self.context.get("weather_factor", 0.7)

                # Base solar gain scaled by daily, seasonal, and weather patterns
                solar_gain = 200.0 * daily_solar * seasonal_factor * weather_factor  # [W] per zone
                return torch.full((batch_size, self.n_zones), solar_gain,
                                  device=self.device, dtype=self.dtype)

            def Q_internal_fn(t, batch_size=1):
                """Context-aware internal heat gains based on occupancy state and schedule."""
                h_of_day = simclock.hour_of_day(t)

                if occupancy_state == "occupied":
                    # Full occupancy schedule
                    if 8 <= h_of_day <= 17:  # Business hours
                        internal_gain = 1200.0  # Full occupancy [W]
                    elif 7 <= h_of_day < 8 or 17 < h_of_day <= 19:  # Transition hours
                        internal_gain = 700.0  # Partial occupancy [W]
                    else:
                        internal_gain = 250.0  # Night: minimal loads [W]
                elif occupancy_state == "unoccupied":
                    # Building unoccupied - minimal equipment loads only
                    internal_gain = 150.0  # [W]
                else:  # "transition" - building startup
                    # Moderate gains during startup period
                    internal_gain = 500.0  # [W]
                return torch.full((batch_size, self.n_zones), internal_gain,
                                  device=self.device, dtype=self.dtype)

            def Q_hvac_fn(t, batch_size=1):
                """Context-aware HVAC heat input based on system mode and schedule."""
                h_of_day = simclock.hour_of_day(t)
                d_of_year = simclock.day_of_year(t)

                # HVAC operation based on occupancy
                if occupancy_state == "unoccupied":
                    hvac_multiplier = 0.1  # Minimal HVAC during unoccupied
                elif occupancy_state == "transition":
                    hvac_multiplier = 0.6  # Moderate HVAC during startup
                else:  # occupied
                    if 7 <= h_of_day <= 19:  # Business hours
                        hvac_multiplier = 1.0  # Full HVAC operation
                    else:
                        hvac_multiplier = 0.4  # Reduced HVAC after hours

                # Base HVAC load depends on system mode
                if system_mode == "cooling":
                    base_hvac = -600.0  # Cooling load [W] (negative)
                elif system_mode == "heating":
                    base_hvac = 800.0  # Heating load [W] (positive)
                elif system_mode == "setback":
                    base_hvac = -200.0  # Minimal cooling for setback
                elif system_mode == "economizer":
                    base_hvac = -300.0  # Reduced mechanical load with free cooling
                else:  # "minimal" or unknown
                    base_hvac = 0.0  # No HVAC load

                # Seasonal bias for realistic operation
                seasonal_bias = -400.0 * torch.sin(torch.tensor(2 * torch.pi * (d_of_year - 80) / 365))  # [W]

                total_hvac = (base_hvac + seasonal_bias) * hvac_multiplier
                return torch.full((batch_size, self.n_zones), total_hvac,
                                  device=self.device, dtype=self.dtype)

            def T_outdoor_fn(t, batch_size=1):
                daily_amplitude = 10.0
                peak_hour = 14.
                seasonal_amplitude = 20.0
                base_temp = T_outdoor_base
                d_of_year = simclock.day_of_year(t)
                h_of_day = simclock.hour_of_day(t)

                daily_temp = daily_amplitude * np.sin(2 * np.pi * (h_of_day - peak_hour) / 24)
                seasonal_temp = seasonal_amplitude * np.sin(2 * np.pi * (d_of_year - 80) / 365)
                total_temp = base_temp + daily_temp + seasonal_temp
                total_temp = torch.full((batch_size, 1), total_temp)
                return total_temp

            self._input_functions = {
                "T_outdoor": T_outdoor_fn,
                "Q_solar": Q_solar_fn,
                "Q_internal": Q_internal_fn,
                "Q_hvac": Q_hvac_fn,
            }

        return self._input_functions