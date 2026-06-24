"""
staged_rtu.py

Discrete, staged packaged rooftop unit (RTU) — the dominant HVAC plant in North
American small commercial buildings (strip malls, retail, small offices): one or
two stages of direct-expansion (DX) cooling with a constant/multi-speed fan,
driven by a thermostat that engages stages in order.

This module deliberately SEPARATES the plant from the controller:

- ``StagedDXPlant``      : given which stages are engaged, produce supply air
                          temperature, flow, and fan/DX power. Memoryless and
                          fully differentiable in its (physical) parameters.
- ``StagingController``  : given the cooling demand, decide which stages engage.
                          Ordinal: stages turn on in order via sequential
                          sigmoids with strictly-positive, ordered thresholds.

Staging is ORDINAL, not categorical: stage 2 only engages on top of stage 1.
We encode that with a cumulative engagement vector ``e`` of length n_stages,
``e[j] in [0, 1]`` = "is stage j+1 engaged". Per-stage capacity, flow, and power
are strictly-positive INCREMENTS, so engaging another stage always adds load on
top of the previous one (monotone by construction).
"""
import torch
import torch.nn as nn

from .base import BuildingComponent
from .._runtime import beartype


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus, for initializing raw (pre-softplus) params."""
    return y + torch.log(-torch.expm1(-y))


@beartype
def stage_to_engagement(stage: torch.Tensor, n_stages: int) -> torch.Tensor:
    """
    Convert an integer stage signal to a cumulative (ordinal) engagement vector.

    stage k -> [1, ..., 1, 0, ..., 0] with the first k entries set, i.e.
    engagement[..., j] = 1 if stage >= j + 1. This is the hard limit of the
    controller's soft sequential-sigmoid output.

    Args:
        stage: integer-valued stage signal, shape [..., 1] (e.g. {0, 1, 2}).
        n_stages: number of DX stages.

    Returns:
        Engagement tensor, shape [..., n_stages], entries in {0, 1}.
    """
    levels = torch.arange(1, n_stages + 1, device=stage.device, dtype=stage.dtype)
    return (stage >= levels).to(stage.dtype)


class StagedDXPlant(BuildingComponent):
    """
    Memoryless staged DX cooling plant for a single packaged RTU.

    Given the engaged stages (and return/outdoor air + occupancy), computes the
    supply air conditions and electrical power. All per-stage quantities are
    strictly-positive increments (softplus-reparameterized), so the model is
    monotone in stage by construction. DX and fan power are modeled with their
    own per-stage increments rather than derived through a single COP, because
    sensible-only cooling and multi-speed fans do not share one efficiency.

    External Inputs (shape [batch, 1] unless noted):
        - stage_engagement: cumulative engagement per stage [0-1], shape [batch, n_stages]
        - T_return:  return air temperature [K] (== zone air for a single-zone RTU)
        - T_outdoor: outdoor dry-bulb temperature [K]
        - occupancy: occupancy signal [0-1] (gates ventilation airflow)

    Outputs (shape [batch, 1]):
        - T_supply:       supply air temperature [K]
        - supply_airflow: supply air mass flow [kg/s]
        - fan_power:      supply fan electrical power [W]
        - dx_power:       compressor (DX) electrical power [W]
        - Q_hvac:         sensible heat delivered to the zone [W] (negative = cooling)
    """

    _state_ranges = {}
    _external_ranges = {
        "stage_engagement": (0.0, 1.0),       # [-] cumulative stage engagement
        "T_return": (283.15, 313.15),         # [K] return (zone) air
        "T_outdoor": (253.15, 323.15),        # [K] outdoor air
        "occupancy": (0.0, 1.0),              # [-] occupancy
    }
    # U/D split: stage engagement is commanded; OAT/occupancy are imposed.
    # T_return is left untagged — it is a fed-back zone (envelope) state, not U or D.
    _control_ranges = {"stage_engagement": (0.0, 1.0)}
    _disturbance_ranges = {
        "T_outdoor": (253.15, 323.15),
        "occupancy": (0.0, 1.0),
    }
    _output_ranges = {
        "T_supply": (273.15, 313.15),         # [K]
        "supply_airflow": (0.0, 3.0),         # [kg/s]
        "fan_power": (0.0, 5000.0),           # [W]
        "dx_power": (0.0, 15000.0),           # [W]
        "Q_hvac": (-30000.0, 5000.0),         # [W] negative = cooling
    }

    @beartype
    def __init__(
        self,
        n_stages: int = 2,
        # Per-stage increments (physical, strictly positive) — sensible defaults
        # informed by the strip-mall dataset.
        dQ_stage: list = (5500.0, 2200.0),       # [W] sensible cooling capacity increment
        dflow_stage: list = (0.07, 0.04),        # [kg/s] airflow increment
        dx_power_stage: list = (4300.0, 2900.0), # [W] DX electrical power increment
        dfan_stage: list = (200.0, 350.0),       # [W] per-stage fan power increment
        # Scalars
        vent_flow: float = 0.95,                 # [kg/s] ventilation airflow when fan on
        fan_power_coeff: float = 300.0,          # [W/(kg/s)^3] cube-law fan power
        dx_oat_coeff: float = 0.02,              # [1/K] DX power rise per K above T_rated
        oa_fraction: float = 0.2,                # [-] outdoor-air fraction for mixing
        T_rated: float = 308.15,                 # [K] OAT rating point for DX power (fixed)
        cp_air: float = 1005.0,                  # [J/kg/K] (fixed)
        learnable: set = None,
        device=None,
        dtype=torch.float32,
        context: dict = None,
    ):
        # Minimal base init (context/device/dtype/n_zones); we manage params here.
        super().__init__(params={"context": context, "n_zones": 1}, learnable=set(),
                         device=device, dtype=dtype)
        self.n_stages = n_stages
        learnable = learnable or set()

        # Positive increment params, stored as raw pre-softplus values.
        self._reg("raw_dQ", _softplus_inv(torch.tensor(dQ_stage)), "dQ_stage" in learnable)
        self._reg("raw_dflow", _softplus_inv(torch.tensor(dflow_stage)), "dflow_stage" in learnable)
        self._reg("raw_dx_power", _softplus_inv(torch.tensor(dx_power_stage)), "dx_power_stage" in learnable)
        self._reg("raw_dfan", _softplus_inv(torch.tensor(dfan_stage)), "dfan_stage" in learnable)
        self._reg("raw_vent_flow", _softplus_inv(torch.tensor(vent_flow)), "vent_flow" in learnable)
        self._reg("raw_fan_coeff", _softplus_inv(torch.tensor(fan_power_coeff)), "fan_power_coeff" in learnable)
        # oa_fraction in [0,1] via sigmoid; dx_oat_coeff unconstrained.
        self._reg("raw_oa", torch.logit(torch.tensor(oa_fraction).clamp(1e-4, 1 - 1e-4)), "oa_fraction" in learnable)
        self._reg("dx_oat_coeff", torch.tensor(dx_oat_coeff), "dx_oat_coeff" in learnable)
        # Fixed references.
        self.register_buffer("T_rated", torch.tensor(T_rated, dtype=self.dtype))
        self.register_buffer("cp_air", torch.tensor(cp_air, dtype=self.dtype))

    def _reg(self, name, value, is_learnable):
        value = value.to(device=self.device, dtype=self.dtype)
        if is_learnable:
            self.register_parameter(name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    # --- positive / bounded physical parameters via reparameterization ---
    @property
    def dQ_stage(self):
        return torch.nn.functional.softplus(self.raw_dQ)

    @property
    def dflow_stage(self):
        return torch.nn.functional.softplus(self.raw_dflow)

    @property
    def dx_power_stage(self):
        return torch.nn.functional.softplus(self.raw_dx_power)

    @property
    def dfan_stage(self):
        return torch.nn.functional.softplus(self.raw_dfan)

    @property
    def vent_flow(self):
        return torch.nn.functional.softplus(self.raw_vent_flow)

    @property
    def fan_power_coeff(self):
        return torch.nn.functional.softplus(self.raw_fan_coeff)

    @property
    def oa_fraction(self):
        return torch.sigmoid(self.raw_oa)

    @beartype
    def forward(
        self, *,
        stage_engagement: torch.Tensor,  # [batch, n_stages] in [0,1]
        T_return: torch.Tensor,          # [batch, 1] K
        T_outdoor: torch.Tensor,         # [batch, 1] K
        occupancy: torch.Tensor,         # [batch, 1] in [0,1]
        t: float = 0.0,
        dt: float = 0.0,
    ) -> dict:
        eps = 1e-6
        eng = stage_engagement  # [batch, n_stages]

        # Fan runs if occupied or any stage engaged; ventilation flow plus per-stage increments.
        fan_on = torch.clamp(occupancy + eng.sum(dim=-1, keepdim=True), 0.0, 1.0)  # [batch,1]
        flow = fan_on * self.vent_flow + (eng * self.dflow_stage).sum(dim=-1, keepdim=True)  # [batch,1]
        mdot_cp = flow * self.cp_air

        # Sensible DX cooling capacity (positive magnitude), cumulative over stages.
        Q_cool = (eng * self.dQ_stage).sum(dim=-1, keepdim=True)  # [batch,1] W

        # Mixed air, then DX cools it. Guard the zero-flow (fan off) case.
        T_mix = self.oa_fraction * T_outdoor + (1.0 - self.oa_fraction) * T_return
        T_supply = torch.where(
            flow > eps,
            T_mix - Q_cool / (mdot_cp + eps),
            T_return,
        )

        # Electrical power: per-stage increments; DX rises with OAT, fan ~ flow^3 + per-stage.
        oat_factor = 1.0 + self.dx_oat_coeff * (T_outdoor - self.T_rated)
        dx_power = (eng * self.dx_power_stage).sum(dim=-1, keepdim=True) * oat_factor
        dx_power = torch.clamp(dx_power, min=0.0)
        fan_power = self.fan_power_coeff * flow ** 3 + (eng * self.dfan_stage).sum(dim=-1, keepdim=True)

        Q_hvac = mdot_cp * (T_supply - T_return)  # heat delivered to zone (negative = cooling)

        return {
            "T_supply": T_supply,
            "supply_airflow": flow,
            "fan_power": fan_power,
            "dx_power": dx_power,
            "Q_hvac": Q_hvac,
        }

    @property
    def input_functions(self):
        """Minimal context-driven inputs so the plant can run standalone.

        Stage engagement defaults to off; the controller supplies it in a closed
        loop. T_return/T_outdoor/occupancy come from context with simple defaults.
        """
        if not hasattr(self, "_input_functions"):
            T_outdoor = self.context.get("T_outdoor", 298.15)
            T_setpoint = self.context.get("T_setpoint_base", 297.15)

            def stage_fn(t, batch_size=1):
                return torch.zeros((batch_size, self.n_stages), device=self.device, dtype=self.dtype)

            def const(value):
                return lambda t, batch_size=1: torch.full((batch_size, 1), float(value),
                                                          device=self.device, dtype=self.dtype)

            self._input_functions = {
                "stage_engagement": stage_fn,
                "T_return": const(T_setpoint),
                "T_outdoor": const(T_outdoor),
                "occupancy": const(1.0),
            }
        return self._input_functions


class StagingController(BuildingComponent):
    """
    Ordinal staging thermostat.

    Maps the cooling demand (return/zone temperature above setpoint) to a
    cumulative stage-engagement vector via sequential sigmoids with
    strictly-positive, ordered thresholds:

        demand   e   = T_return - T_setpoint            (positive => too warm)
        thresh   Θ_k = cumsum(softplus(raw_threshold))  (Θ_1 < Θ_2 < ... > 0)
        engage   e_k = σ((e - Θ_k) / τ)                 (soft, training)
                     = 1[e >= Θ_k]                       (hard, eval)

    Because the thresholds are strictly-positive cumulative increments, stage k+1
    cannot engage before stage k — staging is monotone/ordinal by construction,
    unlike a softmax over stages. The engaged-stage count is ``Σ_k e_k``.

    Discreteness uses a straight-through estimator: the forward value is the hard
    (snapped) engagement so the closed loop sees genuine integer stages, while
    gradients flow through the smooth sigmoid surrogate (so the loop stays
    differentiable for backprop-through-time / controller-in-loop training):

        e = e_soft + (e_hard - e_soft).detach()

    Set ``use_ste=False`` to feed the smooth engagement directly (no snapping).

    Note: this v1 is stateless (no hysteresis). Real thermostats add a deadband
    between stage-up and stage-down thresholds to prevent chatter; that is the
    natural next refinement (carry the current engagement as state and offset the
    stage-down thresholds by a positive deadband).
    """

    _state_ranges = {}
    _external_ranges = {
        "T_return": (283.15, 313.15),    # [K] return (zone) air
        "T_setpoint": (288.15, 305.15),  # [K] cooling setpoint
    }
    # The setpoint is the commanded input; T_return is a fed-back zone state.
    _control_ranges = {"T_setpoint": (288.15, 305.15)}
    _output_ranges = {
        "stage_engagement": (0.0, 1.0),  # [-] cumulative engagement per stage
        "stage": (0.0, 4.0),             # [-] engaged-stage count
    }

    @beartype
    def __init__(
        self,
        n_stages: int = 2,
        stage_thresholds: list = (0.5, 0.7),  # [K] threshold INCREMENTS (Θ via cumsum)
        tau: float = 0.3,                     # [K] sigmoid smoothing width
        use_ste: bool = True,                 # discrete forward, smooth backward
        learnable: set = None,
        device=None,
        dtype=torch.float32,
        context: dict = None,
    ):
        super().__init__(params={"context": context, "n_zones": 1}, learnable=set(),
                         device=device, dtype=dtype)
        self.n_stages = n_stages
        self.use_ste = use_ste
        learnable = learnable or set()

        # Threshold increments and smoothing width: strictly positive via softplus.
        self._reg("raw_thresh", _softplus_inv(torch.tensor(stage_thresholds)),
                  "stage_thresholds" in learnable)
        self._reg("raw_tau", _softplus_inv(torch.tensor(float(tau))), "tau" in learnable)

    def _reg(self, name, value, is_learnable):
        value = value.to(device=self.device, dtype=self.dtype)
        if is_learnable:
            self.register_parameter(name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    @property
    def thresholds(self) -> torch.Tensor:
        """Cumulative, strictly-increasing, strictly-positive stage thresholds [K]."""
        return torch.cumsum(torch.nn.functional.softplus(self.raw_thresh), dim=0)

    @property
    def tau(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_tau)

    @beartype
    def forward(
        self, *,
        T_return: torch.Tensor,    # [batch, 1] K
        T_setpoint: torch.Tensor,  # [batch, 1] K
        t: float = 0.0,
        dt: float = 0.0,
    ) -> dict:
        demand = T_return - T_setpoint                # [batch,1], >0 => need cooling
        gap = demand - self.thresholds                # [batch, n_stages] broadcast
        soft = torch.sigmoid(gap / self.tau)          # smooth surrogate (backward)
        if self.use_ste:
            hard = (gap >= 0.0).to(demand.dtype)      # snapped ordinal staging (forward)
            engagement = soft + (hard - soft).detach()  # straight-through estimator
        else:
            engagement = soft
        stage = engagement.sum(dim=-1, keepdim=True)
        return {"stage_engagement": engagement, "stage": stage}

    @property
    def input_functions(self):
        if not hasattr(self, "_input_functions"):
            T_setpoint = self.context.get("T_setpoint_base", 297.15)

            def const(value):
                return lambda t, batch_size=1: torch.full((batch_size, 1), float(value),
                                                          device=self.device, dtype=self.dtype)

            self._input_functions = {
                "T_return": const(T_setpoint + 1.0),
                "T_setpoint": const(T_setpoint),
            }
        return self._input_functions
