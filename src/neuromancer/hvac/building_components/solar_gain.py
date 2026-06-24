"""
solar_gains.py

Models solar heat gains to building zones from readily available inputs:
outdoor temperature, weather conditions, and basic building specifications.

This model focuses specifically on solar irradiance and heat gains through windows,
avoiding overlap with envelope models that handle conduction and infiltration
through thermal resistance parameters.

Key Features:
- Estimates solar irradiance from outdoor temperature and weather patterns
- Accounts for building orientation and window specifications
- Supports zone vectorization for multi-zone buildings
- Uses empirical correlations based on typical weather relationships
- Compatible with envelope models (no double-counting of envelope heat transfer)
- Inherits from BuildingComponent for simulation and parameter management

Units:
- Temperature: Kelvin [K]
- Power: Watts [W]
- Area: Square meters [m²]
- Irradiance: Watts per square meter [W/m²]
"""
import numpy as np
import torch
import math
from typing import Union, List
from .base import BuildingComponent
import neuromancer.hvac.simclock as simclock
from .._runtime import beartype


class SolarGains(BuildingComponent):
    """
    Models solar heat gains to building zones using outdoor temperature,
    weather conditions, and window specifications.

    This model estimates solar irradiance from readily available weather data
    and calculates resulting heat gains through windows. It's designed to work
    alongside envelope models that handle conduction and infiltration separately.

    Solar Heat Gain Process:
    1. Estimate solar irradiance from outdoor temperature and weather patterns
    2. Account for window orientation relative to sun position
    3. Apply solar heat gain coefficient (SHGC) for window properties
    4. Calculate final solar heat gains delivered to zones

    The model uses empirical relationships to estimate solar irradiance from
    outdoor temperature patterns and weather conditions, making it suitable
    for applications where detailed solar measurement data is unavailable.
    """

    # Variable ranges for BuildingComponent base class
    _external_ranges = {
        "T_outdoor": (253.15, 318.15),  # [K] Outdoor temperature (-20°C to 45°C)
        "weather_factor": (0.0, 1.0),  # [-] Weather clarity (0=overcast, 1=clear)
    }

    _zone_param_ranges = {
        # Zone-specific parameters (expanded to [n_zones] vectors)
        "window_area": (1.0, 100.0),  # [m²] Window area per zone
        "window_orientation": (-180.0, 180.0),  # [deg] Window orientation
        "window_shgc": (0.1, 0.9),  # [-] Solar heat gain coefficient
    }

    _param_ranges = {
        # Shared parameters (scalars)
        "latitude_deg": (-90.0, 90.0),  # [deg] Building latitude
        "max_solar_irradiance": (400.0, 1200.0),  # [W/m²] Peak solar irradiance
    }

    _output_ranges = {
        "Q_solar": (0.0, 10000.0),  # [W] Solar gains per zone
    }

    @beartype
    def __init__(
            self,
            # Scenario/geometry: no defaults (these describe the building and its site)
            n_zones: int,                                    # number of zones
            window_area: Union[float, List[float]],          # [m²] Window area per zone
            window_orientation: Union[float, List[float]],   # [deg] 0=south, 90=west, etc.
            latitude_deg: float,                             # [deg] Building latitude (site)

            # Material / model parameters (defaults OK)
            window_shgc: Union[float, List[float]] = 0.6,    # [-] Solar heat gain coefficient
            max_solar_irradiance: float = 800.0,             # [W/m²] Peak solar irradiance

            # Standard BuildingComponent parameters
            context: dict = None,  # Building operating context (scenario) for init/inputs
            learnable: set = None,
            device=None,
            dtype=torch.float32
    ):
        """
        Initialize solar gains model using BuildingComponent infrastructure.

        Args:
            n_zones: Number of building zones
            window_area: Window area per zone [m²] (scalar or list[n_zones])
            window_orientation: Window orientation [deg] (scalar or list[n_zones])
            window_shgc: Solar heat gain coefficient [0-1] (scalar or list[n_zones])
            latitude_deg: Building latitude [deg]
            max_solar_irradiance: Peak solar irradiance [W/m²]
            learnable: Parameters to make learnable for optimization
            device: Device for tensor computations
            dtype: Tensor data type
        """
        # BuildingComponent handles parameter expansion and device/dtype setup
        super().__init__(params=locals(), learnable=learnable, device=device, dtype=dtype)

        # Convert latitude to radians for calculations
        self.latitude_rad = math.radians(self.latitude_deg)

        # Zone-specific parameters are automatically expanded by base class
        # self.window_area: [n_zones] tensor
        # self.window_orientation: [n_zones] tensor
        # self.window_shgc: [n_zones] tensor

    @beartype
    def estimate_solar_irradiance(
            self,
            t: float,
            T_outdoor: torch.Tensor,
            weather_factor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate solar irradiance from solar geometry, outdoor temperature and weather.

        Solar position depends only on (scalar) time, so the geometry is computed with
        plain math and combined with the per-batch temperature/weather tensors. This
        avoids running datetime through torch.vmap (which is not vmap-compatible).

        Args:
            t: seconds from epoch [s] (scalar).
            T_outdoor: Outdoor air temperature [K], shape [batch_size, 1]
            weather_factor: Weather clarity [0-1], shape [batch_size, 1]

        Returns:
            torch.Tensor: Estimated solar irradiance [W/m²], shape [batch_size, 1]
        """
        # Solar position (scalars, from the shared simulation clock).
        h_of_day = simclock.hour_of_day(t)
        d_of_year = simclock.day_of_year(t)

        day_angle = 2 * math.pi * (d_of_year - 81) / 365
        declination = 23.45 * math.pi / 180 * math.sin(day_angle)
        hour_angle = (h_of_day - 12) * 15 * math.pi / 180

        sin_elevation = (
            math.sin(self.latitude_rad) * math.sin(declination) +
            math.cos(self.latitude_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        solar_elevation = math.asin(max(-1.0, min(1.0, sin_elevation)))

        # Base solar irradiance from geometry (0-d tensor via max_solar_irradiance buffer).
        base_irradiance = self.max_solar_irradiance * max(0.0, math.sin(solar_elevation))

        # Higher outdoor temperatures correlate with stronger solar conditions.
        T_ref = 293.15  # [K] Reference temperature (20°C)
        temp_factor = torch.clamp(1.0 + 0.02 * (T_outdoor - T_ref), 0.5, 1.5)

        estimated_irradiance = base_irradiance * weather_factor * temp_factor
        return torch.clamp(estimated_irradiance, 0.0, self.max_solar_irradiance)

    @beartype
    def calculate_solar_gains(
            self,
            t: float,
            T_outdoor: torch.Tensor,
            weather_factor: torch.Tensor,

    ) -> torch.Tensor:
        """
        Calculate solar heat gains through windows for all zones.

        Args:
            t: seconds from epoch [s] (scalar).
            T_outdoor: Outdoor temperature [K], shape [batch_size, 1]
            weather_factor: Weather clarity [0-1], shape [batch_size, 1]

        Returns:
            torch.Tensor: Solar heat gains [W], shape [batch_size, n_zones]
        """
        # Estimate solar irradiance
        irradiance = self.estimate_solar_irradiance(t, T_outdoor, weather_factor)
        # [batch_size, 1]

        # Calculate solar gains for each zone
        # Simple orientation factor (peak at south, reduced for other orientations)
        orientation_rad = self.window_orientation * math.pi / 180  # [n_zones]
        orientation_factor = torch.clamp(torch.cos(orientation_rad), 0.3, 1.0)  # [n_zones]

        # Solar gains = irradiance × window_area × SHGC × orientation_factor
        Q_solar = (
                irradiance *  # [batch_size, 1]
                self.window_area.unsqueeze(0) *  # [1, n_zones]
                self.window_shgc.unsqueeze(0) *  # [1, n_zones]
                orientation_factor.unsqueeze(0)  # [1, n_zones]
        )  # [batch_size, n_zones]

        return Q_solar

    @beartype
    def forward(
            self, *,
            t: float,  # [s] Time of day
            T_outdoor: torch.Tensor,  # [K] Outdoor temperature, shape [batch_size, 1]
            weather_factor: torch.Tensor,  # [-] Weather clarity, shape [batch_size, 1]
            dt: float = None,  # [s] Time step (unused but required by interface)
    ) -> dict:
        """
        Calculate solar heat gains to building zones.

        Args:
            t: Time of day [s] since midnight
            T_outdoor: Outdoor air temperature [K], shape [batch_size, 1]
            weather_factor: Weather clarity factor [0-1], shape [batch_size, 1]
            dt: Time step [s] (unused but required by BuildingComponent interface)

        Returns:
            dict: Solar gains and diagnostics, all tensors shape [batch_size, n_zones]
                Q_solar: Solar heat gains through windows [W]
        """
        # Solar position depends only on (scalar) time; pass it through directly.
        Q_solar = self.calculate_solar_gains(float(t), T_outdoor, weather_factor)

        self.diagnostics = {}

        return {
            'Q_solar': Q_solar,  # [W] Solar gains through windows
        }

    @property
    def input_functions(self):
        """
        Context-aware input functions for SolarGains component.

        Returns functions that generate realistic solar input patterns based on
        building context for coordinated simulation.

        Returns:
            dict: Mapping from input variable names to callables (t, batch_size) -> torch.Tensor
        """
        if not hasattr(self, '_input_functions'):
            # Get context values with fallbacks

            req_keys = ['T_outdoor', 'weather_factor']
            assert all(key in self.context for key in req_keys), "Context does not contain required keys!"

            T_outdoor_base = self.context.get("T_outdoor")  # Default: 20°C
            weather_factor_base = self.context.get("weather_factor")  # Default: partly cloudy

            def weather_factor_fn(t, batch_size=1):
                """Context-aware weather factor with day/night cycle."""
                # Calculate current hour using context time as baseline
                h_of_day = simclock.hour_of_day(t)

                # Weather factor follows solar availability (zero at night)
                if 6 <= h_of_day <= 18:  # Daylight hours
                    # Use context weather factor during daylight
                    weather_factor = weather_factor_base

                    # Add realistic cloud variation during the day
                    cloud_variation = 0.1 * torch.sin(torch.tensor(2 * torch.pi * (h_of_day - 10) / 8))
                    weather_factor = torch.clamp(
                        weather_factor + cloud_variation, 0.0, 1.0
                    )
                else:
                    # No solar radiation at night
                    weather_factor = torch.tensor(0.0)

                return torch.full((batch_size, 1), weather_factor,
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
                "weather_factor": weather_factor_fn,
            }
        return self._input_functions

    # @input_functions.setter
    # def input_functions(self, value):
    #     """Allow custom input functions to be set."""
    #     if not hasattr(self, '_input_functions'):
    #         self._input_functions = {}
    #     self._input_functions.update(value)