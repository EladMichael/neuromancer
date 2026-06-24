# neuromancer.hvac

A PyTorch-based, **differentiable** building / HVAC physics substrate for grey-box
system identification and control. Every component is an `nn.Module` whose
physical parameters can be marked learnable, so you can fit them to real data by
gradient descent, run them forward as a simulator, or drop them into a
closed-loop control problem.

The package provides the *physics*. It wires into NeuroMANCER's `Node` / `System`
rollout machinery for multi-component buildings, and is meant to be deployed from
a separate experiment/training layer (e.g. an external sys-id framework) that
owns the data adapter and fit harness.

---

## Where things stand

The current focus is a **single-zone envelope + finite-stage RTU** model that can
be fit to a real dataset. Design principles (see [CLAUDE.md](../../../CLAUDE.md)):
KISS, DRY, fail-fast, no speculative abstraction.

- **Zero-order first.** The working models are memoryless where possible; the only
  states are the ones physics demands (zone temperatures). First-order / stateful
  dynamics (supply-air coil lag, controller hysteresis, warmup) are intentionally
  held off and logged in [deferred.md](deferred.md) with the cheapest faithful way
  to add each one later.
- **Runtime type checking** is on by default (beartype) for fail-fast contracts.
  Disable for heavy workloads by setting `RUNTIME_TYPING=0` *before* importing the
  package (see [_runtime.py](_runtime.py)).
- **No "data constructors."** Things that define the building (zone count,
  geometry, RC sizing, time of year) are required, not defaulted. Learnable
  physical parameters keep sensible default initializations.

---

## Core concepts

### BuildingComponent
The base class ([building_components/base.py](building_components/base.py)). A
component declares its variables in partitioned range dicts —
`_state_ranges`, `_external_ranges`, `_output_ranges`, `_param_ranges`,
`_zone_param_ranges` — which drive validation, automatic wiring, and which params
become learnable. Constructor params named in `learnable` become `nn.Parameter`s;
the rest are registered buffers. `forward(t, dt, **inputs)` returns a dict of
named outputs. Components are zone-vectorized (`n_zones`).

Each component can also run **standalone** via `component.simulate(...)`, which
generates its own exogenous inputs and initial state from its context and returns
trajectories shaped `[batch, time, dim]`. This is the quickest physics sanity
check.

### Context
A context ([context.py](context.py)) is a snapshot of operating conditions
("what kind of day are we simulating?") — outdoor temp, time of year, weather
clarity, occupancy/system mode, baseline setpoints. It seeds physically
consistent initial states and the synthetic input functions used for standalone
simulation. Presets: `MILD_COOLING_CONTEXT`, `PEAK_COOLING_CONTEXT`,
`WINTER_HEATING_CONTEXT`, `NIGHT_SETBACK_CONTEXT`, `ECONOMIZER_CONTEXT`,
`TRANSITION_CONTEXT`. When fitting to measured data you seed states from the data
instead, so context matters mainly for autonomous/synthetic runs.

### Wiring: BuildingNode + BuildingSystem
[building.py](building.py) bridges components to NeuroMANCER:

- **`BuildingNode`** exposes a `BuildingComponent` through the same surface as a
  core `neuromancer.system.Node`: an ordered `input_keys` list (the data-dict keys,
  matched positionally to the component's `forward()` arguments — states then
  externals), an `output_keys` list, and `forward(data) -> dict`. It does *not*
  subclass `Node`, because the physics `forward` is keyword-only and strictly
  single-step (no windowed past/future). Outputs default to namespaced
  `"<node>.<key>"` to avoid collisions; pass `output_keys` for bare names (needed
  when a single recurrent node must feed its state back in a plain `System`).
- **`BuildingSystem`** subclasses `SystemPreview` (gets the rollout loop, graph
  viz, and known-future preview for predictive control). It adds initial-condition
  setup from each component's `initial_state_functions` / `input_functions`, a
  `simulate(...)` window helper, and a pure single-step `step(...)`. Components
  connect automatically by matching variable names across nodes; pure exogenous
  inputs get full trajectories, produced variables are seeded at t0 and overwritten
  step by step, and recurrent states are seeded at t0.
- **`CompositeDynamics`** wires `StagingController → StagedDXPlant → Envelope` into
  one recurrent node whose only state is `T_zones` (computing `Q_hvac` internally).
  It is the idiomatic home for the memoryless plant — meaningful only inside a
  state-carrying composite — and drops straight into a core `System` or an external
  integrator via its pure `step(state, controls, disturbances, t, dt)`.

### Units & naming
Internally SI: temperature in Kelvin, power in Watts, airflow in kg/s, pressure
in Pa (gauge), time in seconds from epoch. Variable-name prefixes encode units
(`T_`, `Q_`, `P_`, `t_`, `*_power`, `*_min/_max`, `tau_`). Full convention in
[developer.md](developer.md). Plotting converts to Celsius and labels y-axis
units for display only.

---

## Component catalog

| Component | Role | Notes |
|---|---|---|
| `Envelope` | Multi-zone RC thermal network | RK4 step; consumes raw `irradiance` [W/m²] and `occupancy` [-] and converts them to heat via learnable per-zone `solar_gain` / `internal_gain` (the grey-box "B matrix"); takes `Q_hvac` [W] from the HVAC side. |
| `StagedDXPlant` | Finite-stage (DX) rooftop plant | **The common real-world plant.** Memoryless. Ordinal cumulative engagement over stages; learnable softplus increments for capacity/flow/power per stage. Outputs supply temp, airflow, fan/DX power, `Q_hvac`. |
| `StagingController` | Discrete stage selector | Ordinal threshold on `(setpoint − return temp)` with strictly-positive ordered thresholds (cumsum of softplus); straight-through estimator for discrete-forward / smooth-backward closed-loop BPTT. Only needed closed-loop. |
| `RTU` | Continuous modulating air handler | Damper + coil + fan; gauge supply pressure; bidirectional heat. Used in the multi-component demo. |
| `VAVBox` | VAV terminal unit | Per-zone damper + electric reheat; mode detection on supply-air temp with deadband. |
| `SolarGains` | Geometry-based solar load | Standalone solar model (the envelope's built-in gains supersede it for fitting). |
| `Damper`, `ElectricReheatCoil`, `Actuator` | Actuators | Analytic first-order or instantaneous response. |
| `stage_to_engagement(stage, n_stages)` | Helper | Converts a measured integer stage to the ordinal engagement vector the plant consumes. |

For finite-stage fitting the relevant pair is **`StagedDXPlant`** (plant) +
**`StagingController`** (policy), kept deliberately separate so the plant can be
identified open-loop with the *measured* stage before any controller is involved.

---

## Three ways to use it

### 1. Run one component in isolation
Fastest physics check — no wiring.

```python
from neuromancer.hvac.building_components import Envelope
from neuromancer.hvac.context import MILD_COOLING_CONTEXT

envelope = Envelope(
    n_zones=2,
    R_env=[0.1, 0.12], C_env=[1.2e6, 1.0e6],   # required RC sizing
    R_internal=0.05, adjacency=[[0, 1], [1, 0]],
    context=MILD_COOLING_CONTEXT,
)
results = envelope.simulate(t_duration=86400.0, t_dt=300.0)  # [batch, time, dim]
```

See [examples/single_component_example.py](examples/single_component_example.py).

### 2. Simulate a wired multi-component building
Compose nodes, let the system auto-wire by variable name, roll forward a day.

```python
from neuromancer.hvac.building import BuildingNode, BuildingSystem
# build envelope / rtu / vav components, wrap each in a BuildingNode with an
# ordered input_keys list (matched to forward() args), then:
system = BuildingSystem([rtu_node, vav_node, envelope_node])
results = system.simulate(data=data)   # data carries 't', exogenous trajectories
```

See [examples/hvac_example.py](examples/hvac_example.py) (RTU + VAV + Envelope,
24 h) and [plot.py](plot.py) (`simplot` with unit-aware, Celsius axes).

### 3. Grey-box fitting (system identification)
Mark physical parameters learnable and fit them to data with autograd. Open-loop
**plant identification** feeds the *measured* stage and fits the plant to
reproduce measured outputs — no controller, and since the plant is memoryless
every timestep is independent (one vectorized batch).

```python
from neuromancer.hvac.building_components import StagedDXPlant, stage_to_engagement
plant = StagedDXPlant(n_stages=2, learnable={"dQ_stage", "dflow_stage", ...})
# feed stage_to_engagement(measured_stage), return/outdoor temp, occupancy;
# minimize scaled MSE against measured supply temp / airflow / power.
```

See [examples/fit_staged_rtu.py](examples/fit_staged_rtu.py). The training engine
itself (Problem/Trainer, minibatched rollout windows, val-driven early stopping,
the data adapter) lives in the surrounding experiment layer, not in this package.

---

## Data

[data/](data/) holds the processed Modelica RTU strip-mall dataset (baseline /
excited splits at 5/10/20-min resolution) plus sign-constraint maps under
`data/grad/`. [data/load.py](data/load.py)'s `get_dataset(split, resolution,
dataset)` returns `[1, T, 1]` tensors converted into library units (°C→K, m³/s→kg/s)
and derives `Q_hvac` from supply flow and the supply/zone temperature difference.
See [data/dataset_report.md](data/dataset_report.md).

Dataset roles: **Y** = states/outputs, **U** = control (setpoint, stage),
**D** = disturbances (outdoor temp, irradiance, occupancy).

---

## Layering (who owns what)

- **This package** — differentiable physics substrate: components, wiring,
  contexts, the data module, example fits.
- **NeuroMANCER** — the generic training/rollout engine (`Node`, `System`,
  `Problem`, `Trainer`).
- **Your experiment layer** (e.g. neurotemplate) — the data adapter, role-tagged
  datasets, fit harness, objectives. The adapter that maps a dataset to this
  package's variables is the experiment layer's responsibility, not the library's.

A "domain middle band" (HVAC heads/parameterizations, role-tagged data helpers,
rollout protocol) may migrate into this package over time — driven by real fits
pulling it into existence, not built speculatively.

---

## Install

Dependencies are managed with **uv** (not pip):

```bash
uv add <package>
```

Requires `torch`, `numpy`, `matplotlib`, and `beartype` (runtime type checking).

---

## File map

```
hvac/
├── README_HVAC.md            # this file
├── developer.md              # variable naming + context-init conventions
├── deferred.md               # stateful/first-order dynamics held off (with how-to-add)
├── demo_plan.md              # idiomatic sys-id/control demo sketches (Problem/Trainer)
├── _runtime.py               # central beartype toggle (RUNTIME_TYPING)
├── building.py               # BuildingNode + BuildingSystem + CompositeDynamics (bridge)
├── context.py                # operating-condition presets
├── simclock.py               # epoch-seconds <-> calendar helpers (UTC)
├── plot.py                   # simplot: unit-aware, Celsius plotting
├── building_components/
│   ├── base.py               # BuildingComponent base + expand_parameter
│   ├── envelope.py           # RC thermal envelope w/ learnable solar/internal gains
│   ├── staged_rtu.py         # StagedDXPlant, StagingController, stage_to_engagement
│   ├── rooftop_unit.py       # RTU (continuous modulating air handler)
│   ├── vav_box.py            # VAVBox terminal unit
│   └── solar_gain.py         # SolarGains (geometry-based)
├── actuators/                # Actuator base, Damper, ElectricReheatCoil
├── simulation_inputs/        # schedule / disturbance input functions
├── data/                     # processed RTU dataset + load.py + report
└── examples/
    ├── single_component_example.py   # standalone envelope
    ├── hvac_example.py               # wired RTU + VAV + Envelope, 24 h
    └── fit_staged_rtu.py             # open-loop plant identification
```
