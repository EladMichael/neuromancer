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
import neuromancer.hvac.simclock as simclock
from .base import BuildingComponent
from .._runtime import beartype


def rk4_step(deriv, y, dt, n_substeps=1):
    """
    Fixed-step classical RK4 integration of dy/dt = deriv(y) over a span `dt`.

    Exogenous inputs are held constant across the span (zero-order hold), so `deriv`
    is a pure function of the state. For building RC dynamics the step `dt` (minutes)
    is far below the thermal time constants (hours), so a single substep is accurate;
    increase `n_substeps` if integrating with very large `dt`.

    Args:
        deriv: callable(y) -> dy/dt, same shape as y.
        y: state tensor.
        dt: total integration span [s].
        n_substeps: number of equal RK4 substeps over the span.

    Returns:
        State tensor after `dt`.
    """
    h = dt / n_substeps
    for _ in range(n_substeps):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * h * k1)
        k3 = deriv(y + 0.5 * h * k2)
        k4 = deriv(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


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
        - irradiance: Solar irradiance [W/m^2] (driving signal; -> solar heat via solar_gain)
        - occupancy: Occupancy signal [-] (driving signal; -> internal heat via internal_gain)
        - Q_hvac: HVAC heat input/output per zone [W] (negative=cooling, positive=heating)

    Zone-Specific Parameters (expandable to [n_zones] vectors):
        - R_env: Envelope thermal resistance per zone [K/W]
        - C_env: Thermal capacitance per zone [J/K]
        - solar_gain: Irradiance->Watts coupling per zone [W/(W/m^2)] = effective aperture (area*SHGC)
        - internal_gain: Occupancy->Watts coupling per zone [W per unit occupancy]

    Shared Parameters (scalars):
        - R_internal: Inter-zone thermal resistance [K/W]
        - adjacency_threshold: Threshold for discretizing learned topology [0-1]

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
        - Internal fixed-step RK4 integration handles temperature dynamics

    Integration:
        - Fixed-step classical RK4 over each forward() step (see rk4_step / rk4_substeps)
        - Exogenous inputs held constant across the step (zero-order hold)
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
        "irradiance": (0.0, 1200.0),  # [W/m^2] Solar irradiance driving solar gain
        "occupancy": (0.0, 1.0),  # [-] Occupancy signal driving internal gain
        "Q_hvac": (-5000.0, 5000.0),  # [W] HVAC heat/cool addition per zone
    }
    # U/D split of the externals (subset tags; the union above is unchanged).
    _disturbance_ranges = {
        "T_outdoor": (253.15, 318.15),
        "irradiance": (0.0, 1200.0),
        "occupancy": (0.0, 1.0),
    }
    _control_ranges = {"Q_hvac": (-5000.0, 5000.0)}  # HVAC actuation
    _zone_param_ranges = {
        # Zone-specific parameters (expanded to [n_zones] vectors)
        "R_env": (0.05, 2.0),  # [K/W] Envelope resistance per zone
        "C_env": (1e5, 5e7),   # [J/K] Envelope capacitance per zone
        # Disturbance->heat couplings (the grey-box "B matrix"), per zone. These map
        # the raw measured signals to Watts; you never measure gains in Watts directly.
        "solar_gain": (0.0, 50.0),     # [W/(W/m^2)] effective aperture (area * SHGC)
        "internal_gain": (0.0, 5000.0),  # [W per unit occupancy]
    }
    _param_ranges = {
        # Shared parameters (scalars)
        "R_internal": (0.01, 1.0),      # [K/W] Inter-zone resistance
        "adjacency_threshold": (0.0, 1.0),  # [0-1] Threshold for adjacency
    }

    # Number of fixed RK4 substeps per forward() step. 1 is accurate for typical
    # building dt (minutes) vs thermal time constants (hours); override per-instance
    # if you integrate with very large dt.
    rk4_substeps = 1

    @property
    def state_widths(self) -> dict:
        """T_zones is the per-zone temperature vector, width n_zones."""
        return {"T_zones": self.n_zones}

    @beartype
    def __init__(
            self,
            n_zones: int,  # building geometry: number of zones (no default)
            # Zone-specific parameters (can be scalar or list[n_zones])
            R_env: float | list = 0.1,     # [K/W] Envelope resistance per zone
            C_env: float | list = 1e6,     # [J/K] Thermal capacitance per zone
            # Disturbance->heat couplings (grey-box B), per zone, learnable
            solar_gain: float | list = 3.0,        # [W/(W/m^2)] effective aperture (area*SHGC)
            internal_gain: float | list = 800.0,   # [W per unit occupancy]
            # Shared parameters
            R_internal: float | list | np.ndarray | torch.Tensor = 0.02,  # [K/W] Inter-zone resistances
            adjacency_threshold: float = 0.5,  # [0-1] Adjacency threshold
            # Special adjacency matrix parameter
            adjacency: list | np.ndarray | torch.Tensor = None,  # [n_zones, n_zones] Zone connectivity
            # Standard parameters
            context: dict = None,  # Building operating context (scenario) for init/inputs
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
        Get the (n_zones, n_zones) inter-zone resistance matrix.

        Stored as log-resistances so the exponential guarantees positivity. The
        diagonal is irrelevant to the dynamics (the i==i temperature difference is
        zero and the adjacency diagonal is zero), so it is returned as-is.

        Returns:
            Tensor: Resistance matrix (n_zones, n_zones), all entries positive [K/W].
        """
        # Exponential ensures positive resistances; preserves the full per-pair matrix.
        return torch.exp(self.R_internal_logits)

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
    def _dT_dt(
            self,
            T_zones: torch.Tensor,  # [K] Zone temperatures, shape [batch_size, n_zones]
            T_outdoor: torch.Tensor,  # [K] Outdoor temperature, [batch_size, 1] or [batch_size, n_zones]
            irradiance: torch.Tensor,  # [W/m^2] Solar irradiance, [batch_size, 1] or [batch_size, n_zones]
            occupancy: torch.Tensor,  # [-] Occupancy signal, [batch_size, 1] or [batch_size, n_zones]
            Q_hvac: torch.Tensor,  # [W] HVAC input, shape [batch_size, n_zones]
    ) -> torch.Tensor:
        """
        Zone temperature derivatives for RC-network thermal dynamics.

        Pure function of the state given the (held-constant) exogenous inputs, so it
        can be called repeatedly by the RK4 integrator. Includes:
        - Heat exchange with ambient through envelope resistance
        - Inter-zone heat transfer through the internal resistance matrix
        - Solar/internal heat from measured signals via learnable per-zone gains
        - HVAC heat input

        Returns:
            Tensor: Temperature derivatives [K/s], shape [batch_size, n_zones].
        """
        # Measured disturbances -> heat via learnable per-zone gains (grey-box B).
        # Clamp the gains non-negative: irradiance and occupancy can only add heat.
        # Broadcasting: [batch_size, 1] * [n_zones] -> [batch_size, n_zones]
        Q_solar = irradiance * self.solar_gain.clamp(min=0.0)
        Q_internal = occupancy * self.internal_gain.clamp(min=0.0)

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
            irradiance: torch.Tensor,  # [W/m^2] Solar irradiance, [batch_size, 1]
            occupancy: torch.Tensor,  # [-] Occupancy signal, [batch_size, 1]
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
            irradiance (Tensor): Solar irradiance [W/m^2], shape [batch_size, 1] or [batch_size, n_zones].
            occupancy (Tensor): Occupancy signal [-], shape [batch_size, 1] or [batch_size, n_zones].
            Q_hvac (Tensor): HVAC heat input/output [W], shape [batch_size, n_zones].
            dt (float): Time step [s] for integration.

        Returns:
            dict: Updated zone temperatures and diagnostics, all tensors shape [batch_size, n_zones].
                T_zones: Zone temperatures after dynamics integration [K]
        """
        # Exogenous inputs are held constant across the step (zero-order hold), so the
        # derivative is a pure function of the zone temperatures for the RK4 integrator.
        def deriv(T):
            return self._dT_dt(T, T_outdoor, irradiance, occupancy, Q_hvac)

        T_new = rk4_step(deriv, T_zones, dt, n_substeps=self.rk4_substeps)
        return {
            "T_zones": T_new,
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

            def irradiance_fn(t, batch_size=1):
                """Context-aware solar irradiance [W/m^2] with daily/seasonal pattern."""
                h_of_day = simclock.hour_of_day(t)
                d_of_year = simclock.day_of_year(t)
                # Daily solar pattern (zero at night, peak at solar noon)
                if 6 <= h_of_day <= 18:  # Daylight hours
                    daily_solar = torch.sin(torch.tensor(torch.pi * (h_of_day - 6) / 12))  # Peak at noon
                else:
                    daily_solar = torch.tensor(0.0)

                # Seasonal variation (stronger in summer, weaker in winter)
                seasonal_factor = 1.0 + 0.5 * torch.sin(torch.tensor(2 * torch.pi * (d_of_year - 80) / 365))

                # Weather factor from context affects clear-sky fraction
                weather_factor = self.context.get("weather_factor", 0.7)

                # Peak clear-sky irradiance ~900 W/m^2 scaled by daily/seasonal/weather
                irradiance = 900.0 * daily_solar * seasonal_factor * weather_factor  # [W/m^2]
                return torch.full((batch_size, 1), float(irradiance),
                                  device=self.device, dtype=self.dtype)

            def occupancy_fn(t, batch_size=1):
                """Context-aware occupancy signal [0-1] from the building schedule."""
                h_of_day = simclock.hour_of_day(t)
                if occupancy_state == "occupied":
                    occ = 1.0 if 8 <= h_of_day <= 18 else 0.0
                elif occupancy_state == "unoccupied":
                    occ = 0.0
                else:  # "transition" - partial
                    occ = 0.5
                return torch.full((batch_size, 1), occ,
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
                return torch.full((batch_size, 1), float(total_temp),
                                  device=self.device, dtype=self.dtype)

            self._input_functions = {
                "T_outdoor": T_outdoor_fn,
                "irradiance": irradiance_fn,
                "occupancy": occupancy_fn,
                "Q_hvac": Q_hvac_fn,
            }

        return self._input_functions