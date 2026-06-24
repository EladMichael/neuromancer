"""
Bridge between the BuildingComponent interface and NeuroMANCER's Node/System
architecture, so building components benefit from automatic wiring, trajectory
storage, and graph visualization for end-to-end building HVAC simulation.

Rather than re-implementing the rollout machinery, this module reuses
``neuromancer.system.Node`` and ``neuromancer.system.SystemPreview`` directly:

- ``BuildingNode`` adapts a ``BuildingComponent`` (keyword forward(), prefixed
  outputs, scalar t/dt) to the Node interface.
- ``BuildingSystem`` subclasses ``SystemPreview`` (which adds preview of future
  known variables, useful for predictive control) and adds building-specific
  conveniences: initial-condition setup from each component's
  ``initial_state_functions`` / ``input_functions``, and a simple ``simulate``
  window helper whose start time comes from the component context.

Initial conditions come from context-driven initialization (see context.py and
``self.context`` on BuildingComponent). Components start from physically
consistent states/inputs, which keeps startup transients small.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from neuromancer.system import Node, SystemPreview
from neuromancer.hvac.building_components.base import BuildingComponent
from ._runtime import beartype


class BuildingNode(nn.Module):
    """
    Adapter that exposes a BuildingComponent through the same surface as a core
    ``neuromancer.system.Node``: an ordered ``input_keys`` list, an ``output_keys``
    list, a ``name``, a ``callable``, and ``forward(data) -> dict``.

    It does **not** subclass ``Node`` because a building component's ``forward`` is
    keyword-only, dict-returning, and strictly single-step: it consumes the
    *current* value of each variable, never the windowed past/future blocks that
    data-driven nodes can. So ``input_keys`` are mapped onto the component's
    declared keyword arguments by position (states first, then externals), and
    ``t``/``dt`` are threaded per step as Python scalars.

    Outputs default to the namespaced ``"<name>.<key>"`` form so that several
    components can emit the same physical variable (two RTUs producing ``Q_hvac``)
    without colliding. Pass ``output_keys`` to choose your own (e.g. bare names).
    """
    def __init__(self,
                 component: BuildingComponent,
                 input_keys: List[str],
                 output_keys: Optional[List[str]] = None,
                 name: Optional[str] = None):
        """
        Args:
            component: Instance of BuildingComponent to wrap.
            input_keys: Data-dict keys, ordered to match the component's forward()
                arguments (``_state_ranges`` first, then ``_external_ranges``).
                ``t``/``dt`` are threaded automatically and must not be listed.
            output_keys: Data-dict keys to emit, ordered to match the component's
                produced variables (``_output_ranges`` then ``_state_ranges``,
                deduplicated). Defaults to namespaced ``"<name>.<key>"``.
            name: Node name (used for graph viz and the default output namespace).
        """
        super().__init__()
        name = name or f"{type(component).__name__}_{id(component)}"

        # Component forward() keyword args, in declared order: states then externals.
        self._arg_names = list(component._state_ranges) + list(component._external_ranges)
        input_keys = list(input_keys)
        assert len(input_keys) == len(self._arg_names), (
            f"BuildingNode '{name}' expects one input key per forward() argument, "
            f"in order {self._arg_names}; got {input_keys}"
        )
        # data-dict key -> component keyword argument
        self._key_to_arg = dict(zip(input_keys, self._arg_names))

        # Component output dict keys, in declared order (outputs then states),
        # deduplicated (a state may also be an output, e.g. Envelope.T_zones).
        out_names = list(dict.fromkeys(
            list(component._output_ranges) + list(component._state_ranges)
        ))
        if output_keys is None:
            output_keys = [f'{name}.{k}' for k in out_names]
        else:
            output_keys = list(output_keys)
            assert len(output_keys) == len(out_names), (
                f"BuildingNode '{name}' expects one output key per produced variable, "
                f"in order {out_names}; got {output_keys}"
            )
        # component output name -> emitted data-dict key
        self._out_map = dict(zip(out_names, output_keys))

        # t/dt are threaded per rollout step like any other input key.
        self.input_keys = input_keys + ['t', 'dt']
        self.output_keys = output_keys
        self.component = component
        self.callable, self.name = component, name

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adapt the Node (named-key) interface to the BuildingComponent (keyword)
        interface. ``t`` and ``dt`` are passed as Python floats.
        """
        kwargs = {arg: data[k] for k, arg in self._key_to_arg.items()}
        kwargs['t'] = float(data['t'].reshape(-1)[0])
        kwargs['dt'] = float(data['dt'].reshape(-1)[0])
        outputs = self.component(**kwargs)
        return {self._out_map.get(k, f'{self.name}.{k}'): v for k, v in outputs.items()}

    def __repr__(self):
        return f"BuildingNode({self.name}: {', '.join(self.input_keys)} -> {', '.join(self.output_keys)})"


class BuildingSystem(SystemPreview):
    """
    Building-specific extension of SystemPreview.

    Inherits the rollout loop, graph visualization, and future-variable preview
    from SystemPreview, and adds:
    - Building-appropriate defaults (nstep_key='t').
    - Initial-condition setup from BuildingComponent methods.
    - A convenience ``simulate`` window helper.
    """

    def __init__(self,
                 nodes: List[BuildingNode],
                 **kwargs):
        kwargs.setdefault('nstep_key', 't')
        kwargs.setdefault('name', "BuildingSystem")
        super().__init__(nodes, **kwargs)

    @beartype
    def forward(self, input_dict: dict[str, torch.Tensor]):
        data = self.setup(input_dict.copy())
        return super().forward(data)

    def setup(self, data):
        """
        Initialize every component input for a well-posed rollout:

        - Recurrent states are seeded at t0 from ``initial_state_functions``; the
          owning node produces all subsequent steps.
        - Variables produced by some node are seeded at t0 from that component's
          ``input_functions`` (the producer overwrites them step by step).
        - Pure exogenous inputs (no producing node) get a full trajectory over
          the whole horizon from ``input_functions``.

        ``t`` and ``dt`` must already be present (see ``simulate``).
        """
        data = data.copy()
        assert 't' in data, "Time 't' must be in data dict"
        assert 'dt' in data, "Time step 'dt' must be in data dict"

        batch_size = data['t'].shape[0]
        nsteps = data['t'].shape[1]
        t_vec = data['t']  # [batch, nsteps, 1]

        building_nodes = [n for n in self.nodes if isinstance(n, BuildingNode)]
        produced = set().union(*[set(n.output_keys) for n in building_nodes]) if building_nodes else set()

        for node in building_nodes:
            for k in node.input_keys:
                if k in data or k in ('t', 'dt'):
                    continue
                arg = node._key_to_arg[k]
                component = node.component
                if arg in component._state_ranges:
                    init0 = component.initial_state_functions()[arg](batch_size)  # [batch, dim]
                    data[k] = init0.unsqueeze(1)
                elif k in produced:
                    t0 = float(t_vec[0, 0, 0])
                    data[k] = component.input_functions[arg](t0, batch_size).unsqueeze(1)
                else:
                    fn = component.input_functions[arg]
                    steps = [fn(float(t_vec[0, i, 0]), batch_size) for i in range(nsteps)]
                    data[k] = torch.stack(steps, dim=1)  # [batch, nsteps, dim]
        return data

    @beartype
    def simulate(
        self,
        data: Dict = None,
        t_dt: float = 300.0,        # 5 minutes in seconds
        t_duration: float = 86400.0,  # one day in seconds
        t_start: float = None,      # [s from epoch]; defaults to context["t_context"]
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper around ``forward`` that builds the time/time-step
        arrays for a window and runs the rollout.

        With ``data`` provided, it must contain 't' (a 'dt' array is added if
        missing). Otherwise a window is generated; ``t_start`` defaults to the
        component context's ``t_context`` (time of year/day), not a magic number.
        """
        if data is None:
            if t_start is None:
                t_start = self._context_t_start()
            times = torch.arange(t_start, t_start + t_duration, t_dt)
            nsteps = times.shape[0]
            data = {
                't': times.reshape(1, -1, 1).expand(batch_size, -1, 1).clone(),
                'dt': torch.full((batch_size, nsteps, 1), float(t_dt)),
            }
        else:
            assert 't' in data, "If passing in data to simulate, must include 't'"
            if 'dt' not in data:
                data['dt'] = torch.full_like(data['t'], float(t_dt))

        return self.forward(data)

    def _context_t_start(self) -> float:
        """Resolve simulation start time from the first component context."""
        for node in self.nodes:
            component = getattr(node, 'component', None)
            context = getattr(component, 'context', None)
            if context and 't_context' in context:
                return context['t_context']
        raise ValueError(
            "t_start was not provided and no component context with 't_context' was found"
        )

    def step(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run every node once over a single-step (2-D) data dict and return the
        inputs merged with all node outputs. Pure: unlike ``forward``, it does no
        context-driven input synthesis (no ``setup``) and no multi-step rollout,
        so an external integrator can own the loop without a nested rollout.

        ``data`` must already provide every node input (including ``t``/``dt``) as
        ``[batch, dim]`` tensors. Nodes run in list order, so producers must
        precede consumers.
        """
        out = dict(data)
        for node in self.nodes:
            out.update(node.forward(out))
        return out


class CompositeDynamics(nn.Module):
    """
    Compose a staging controller, a memoryless plant, and a stateful envelope into
    ONE dynamics block whose only state is the zone-temperature vector ``T_zones``.

    This is the idiomatic home for a memoryless component like ``StagedDXPlant``:
    it is an algebraic coupling, meaningful only inside a state-carrying composite
    (Controller -> Plant -> Envelope), never as a standalone state-transition model.
    The composite computes ``Q_hvac`` internally, so callers supply only raw
    controls (setpoint) and disturbances (outdoor temp, irradiance, occupancy).

    It mirrors the core ``Node`` surface (``input_keys`` / ``output_keys`` /
    ``forward(data) -> dict``) so it drops straight into ``neuromancer.system.System``
    as a single recurrent node whose output ``T_zones`` is fed back as input. It
    also exposes a pure ``step`` for an external integrator.

    Single-zone: the plant's return air is the zone air, so ``n_zones == 1``.
    """
    def __init__(self,
                 controller: BuildingComponent,
                 plant: BuildingComponent,
                 envelope: BuildingComponent,
                 state_key: str = 'T_zones',
                 control_keys=('T_setpoint',),
                 disturbance_keys=('T_outdoor', 'irradiance', 'occupancy'),
                 name: str = 'hvac_dynamics'):
        super().__init__()
        assert envelope.n_zones == 1, (
            "CompositeDynamics is single-zone (plant return air == zone air); "
            f"got envelope.n_zones={envelope.n_zones}"
        )
        self.controller = controller
        self.plant = plant
        self.envelope = envelope
        self.state_key = state_key
        self.control_keys = list(control_keys)
        self.disturbance_keys = list(disturbance_keys)
        self.name = name
        self.input_keys = [state_key] + self.control_keys + self.disturbance_keys + ['t', 'dt']
        self.output_keys = [state_key]

    def step(self, state: Dict[str, torch.Tensor],
             controls: Dict[str, torch.Tensor],
             disturbances: Dict[str, torch.Tensor],
             t: float, dt: float):
        """
        One pure dynamics step. Returns ``(next_state, outputs)`` where
        ``next_state`` carries ``state_key`` and ``outputs`` carries the controller
        and plant signals (stage, supply temp, powers, ``Q_hvac``) for inspection
        or objectives.
        """
        T_zones = state[self.state_key]
        T_return = T_zones  # single-zone RTU: return air is the zone air

        ctrl_out = self.controller(
            T_return=T_return, T_setpoint=controls['T_setpoint'], t=t, dt=dt,
        )
        plant_out = self.plant(
            stage_engagement=ctrl_out['stage_engagement'],
            T_return=T_return,
            T_outdoor=disturbances['T_outdoor'],
            occupancy=disturbances['occupancy'],
            t=t, dt=dt,
        )
        env_out = self.envelope(
            T_zones=T_zones,
            T_outdoor=disturbances['T_outdoor'],
            irradiance=disturbances['irradiance'],
            occupancy=disturbances['occupancy'],
            Q_hvac=plant_out['Q_hvac'],
            t=t, dt=dt,
        )
        next_state = {self.state_key: env_out['T_zones']}
        outputs = {**ctrl_out, **plant_out, **env_out}
        return next_state, outputs

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single recurrent step for ``System`` rollout: reads the current state +
        controls + disturbances, returns the next ``T_zones``."""
        t = float(data['t'].reshape(-1)[0])
        dt = float(data['dt'].reshape(-1)[0])
        state = {self.state_key: data[self.state_key]}
        controls = {k: data[k] for k in self.control_keys}
        disturbances = {k: data[k] for k in self.disturbance_keys}
        next_state, _ = self.step(state, controls, disturbances, t, dt)
        return next_state
