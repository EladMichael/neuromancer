# TODO: Fix plotting
# TODO: Confirm single component simulations still work.
"""
Building System Simulation Example

This example demonstrates a complete building HVAC system simulation using:
- RTU (Rooftop Unit) as central air handler
- VAVBox terminal units for 2 zones
- 2-zone Envelope for thermal dynamics
- SolarGains for external heat gains

The components are automatically wired together based on matching variable names.

System Architecture:
SolarGains → Envelope ← VAVBox ← RTU
               ↓         ↑       ↑
           T_zones → setpoint   return_airflow_zones
"""

import torch

# Import building components
from neuromancer.hvac.building_components import RTU, VAVBox, Envelope, SolarGains
from neuromancer.hvac.building import BuildingNode, BuildingSystem
from neuromancer.hvac.plot import simplot

print("Building System Simulation Example")
print("="*60)
print("Components: RTU + VAVBox + Envelope + SolarGains")
print("Configuration: 2 zones, automatic wiring")
print("="*60)

# System configuration
n_zones = 2

# =============================================================================
# CREATE BUILDING COMPONENTS
# =============================================================================

print("Creating building components...")

# 1. Solar gains for external heat input
solar = SolarGains(
    n_zones=n_zones,
    window_area=25.0,  # m² per zone
    window_orientation=[0.0, 90.0],  # South, West facing windows
    window_shgc=0.6,   # Solar heat gain coefficient
    latitude_deg=40.0, # Building latitude
    max_solar_irradiance=800.0  # W/m²
)

# 2. Building envelope for thermal dynamics
envelope = Envelope(
    n_zones=n_zones,
    R_env=[0.1, 0.12],    # Zone-specific thermal resistance [K/W]
    C_env=[1.2e6, 1.0e6], # Zone-specific thermal mass [J/K]
    R_internal=0.05,      # Inter-zone resistance [K/W]
    adjacency=[[0.0, 1.0], [0.0, 1.0]],
)

# 3. RTU central air handler
rtu = RTU(
    n_zones=n_zones,
    airflow_max=4.0,      # Total system capacity [kg/s]
    airflow_oa_min=0.4,      # Minimum outdoor air [kg/s]
    Q_coil_max=20000,     # Heating/cooling capacity [W]
    fan_power_per_flow=800,  # Fan efficiency [W/(kg/s)]
    cooling_COP=3.2,      # Cooling efficiency
    heating_efficiency=0.88  # Heating efficiency
)

# 4. VAV boxes for zone control
vav = VAVBox(
    n_zones=n_zones,
    airflow_min=[0.1, 0.08],     # Zone minimums [kg/s]
    airflow_max=[0.8, 0.6],      # Zone maximums [kg/s]
    control_gain=[2.5, 2.0],     # Zone control sensitivity
    Q_reheat_max=[3000, 2500], # Zone reheat capacity [W]
    reheat_efficiency=0.95       # Electric reheat efficiency
)

# =============================================================================
# CREATE BUILDING SYSTEM NODES
# =============================================================================

print("Creating BuildingNode wrappers...")

# Wrap components as nodes
envelope_inputs = {
    "envelope.T_zones": "T_zones",
    "T_outdoor": "T_outdoor",
    "solar.Q_solar": "Q_solar",
    "Q_internal": "Q_internal",
    "vav.Q_supply_flow": "Q_hvac"
}

rtu_inputs = {
    "T_outdoor": "T_outdoor",
    "envelope.T_zones": "T_return_zones",
    "vav.supply_airflow": "return_airflow_zones",
    "rtu_T_supply_setpoint": "T_supply_setpoint",
    "rtu_supply_airflow_setpoint": "supply_airflow_setpoint",
    "rtu.damper_position": "damper_position",
    "rtu.valve_position": "valve_position",
    "rtu.T_supply": "T_supply",
    "rtu.integral_accumulator": "integral_accumulator",
}

vav_inputs = {
    "envelope.T_zones": "T_zone",
    "vav_T_setpoint": "T_setpoint",
    "rtu.T_supply": "T_supply_upstream",
    "rtu.P_supply": "P_duct",
    "vav.damper_position": "damper_position",
    "vav.reheat_position": "reheat_position",
}

solar_inputs = {
    "T_outdoor": "T_outdoor",
    "weather_factor": "weather_factor",
    "day_of_year": "day_of_year"
}

solar_node = BuildingNode(solar, input_map=solar_inputs, name="solar")
envelope_node = BuildingNode(envelope, input_map=envelope_inputs, name="envelope")
rtu_node = BuildingNode(rtu, input_map=rtu_inputs, name="rtu")
vav_node = BuildingNode(vav, input_map=vav_inputs, name="vav")

# 1. SolarGains - generates solar_gains (external input)
# 2. RTU - processes return air, generates supply conditions
# 3. VAVBox - modulates supply air, generates zone loads
# 4. Envelope - integrates all heat sources, updates zone temperatures

# =============================================================================
# RUN BUILDING SIMULATION
# =============================================================================

print("\nRunning 24-hour simulation...")
results = envelope.simulate(
    duration_hours=24.0,    # Full day
    dt_minutes=5.0,         # 5-minute time steps
    t_start_hour=6.0,       # Start at 6 AM
    batch_size=1,
)
print(f"Simulation complete!")
print(f"Results contain {len(results)} variables")
print(f"Variables: {list(results.keys())}")


fig, _ = simplot(
    vav,
    results=results,
    figsize=(14, 10),
    title="24 Hour Simulation",
    filename='plots/24hr_vav.png'
)