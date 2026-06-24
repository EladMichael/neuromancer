"""
Single-component simulation example.

Demonstrates running one BuildingComponent in isolation. Each component can
generate its own exogenous inputs and initial state from its context, so
``component.simulate(...)`` produces a self-contained trajectory without any
wiring. This is the simplest way to sanity-check a component's physics.

Scenario values (zone count, geometry, time of year) live here / in the
component context, not as constructor defaults.
"""
import torch

from neuromancer.hvac.building_components import Envelope
from neuromancer.hvac.context import MILD_COOLING_CONTEXT

n_zones = 2

# Building envelope for thermal dynamics. RC values keep sensible defaults;
# zone count and connectivity describe this particular building.
envelope = Envelope(
    n_zones=n_zones,
    R_env=[0.1, 0.12],       # [K/W] Zone-specific envelope resistance
    C_env=[1.2e6, 1.0e6],    # [J/K] Zone-specific thermal mass
    R_internal=0.05,         # [K/W] Inter-zone resistance
    adjacency=[[0.0, 1.0], [1.0, 0.0]],  # the two zones share a wall
    # Standalone envelope free-runs on its own synthetic (toy) HVAC, so it is not
    # energy-balanced — use a mild scenario to keep it illustrative, not realistic.
    context=MILD_COOLING_CONTEXT,
)

# t_start defaults to the context's time of year/day (no magic numbers).
results = envelope.simulate(
    t_duration=86400.0,  # 24 hours [s]
    t_dt=300.0,          # 5-minute steps [s]
)

T_zones = results["T_zones"]  # [batch, time, n_zones]
print("Single-component Envelope simulation complete.")
print(f"Trajectory shape [batch, time, zones]: {tuple(T_zones.shape)}")
for z in range(n_zones):
    zone = T_zones[0, :, z] - 273.15  # K -> C
    print(f"  Zone {z + 1}: min {zone.min():.1f} C, max {zone.max():.1f} C")

# Plotting (optional): from neuromancer.hvac.plot import simplot
#   simplot(envelope, results=results, title="Envelope - 24h")
