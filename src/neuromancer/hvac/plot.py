"""
plot.py

Standalone plotting function that works with both BuildingComponent and BuildingSystem objects.
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from typing import Dict, List, Tuple, Optional, Union
from math import floor
from neuromancer.hvac.building_components.base import BuildingComponent
from neuromancer.hvac.building import BuildingSystem

from ._runtime import beartype


@beartype
def simplot(
        model: Union['BuildingComponent', 'BuildingSystem'],
        results: Optional[Dict[str, torch.Tensor]] = None,
        # Plotting-specific parameters
        variables: Optional[List[str]] = None,  # List of variables to plot, None = all
        time_range: Optional[Tuple[float, float]] = None,  # (start_hour, end_hour) for zooming
        filename: Optional[str] = None,  # Save figure to this file
        figsize: Optional[Tuple[float, float]] = None,  # Figure size (width, height)
        title: Optional[str] = None,  # Custom title
        batch_idx: int = 0,  # Which batch element to plot

        # Simulation parameters (if results not provided)
        **kwargs
) -> Tuple[plt.Figure, Dict[str, torch.Tensor]]:
    """
    Universal plotting function for BuildingComponent and BuildingSystem objects.

    With standardized interfaces (both use .simulate() and return [batch_size, time, dim]),
    no type checking is needed!

    Args:
        model: BuildingComponent or BuildingSystem instance to plot
        results (dict, optional): Pre-computed simulation results. If None, runs model.simulate().
        variables (list, optional): List of variable names to plot. If None, plots all.
        time_range (tuple, optional): (start_hour, end_hour) to zoom into specific period.
        filename (str, optional): Save figure to this file.
        figsize (tuple, optional): Figure size (width, height). Auto-calculated if None.
        title (str, optional): Custom title. Auto-generated if None.
        batch_idx (int): Which batch element to plot (default: 0).

        **kwargs: Simulation parameters passed to model.simulate().

    Returns:
        tuple: (fig, results) - matplotlib figure and simulation results dict

    Example usage:
        # Works identically for both types!
        fig, results = plot_building_simulation(envelope, duration_hours=24)
        fig, results = plot_building_simulation(system, duration_hours=24)
    """
    model_name = getattr(model, 'name', model.__class__.__name__)
    
    if results is None:
        # Run the model. simulate() uses t_dt (seconds) and resolves t_start from
        # the component context when not given.
        sim_kwargs = {'t_duration': 86400.0, 't_dt': 300.0, **kwargs}
        results = model.simulate(**sim_kwargs)

    # Time parameters are read from the results' absolute time vector either way.
    t_start, dt, t_duration = _extract_time_params_from_results(results, batch_idx)

    var_names = _select_variables(results, variables)

    fig = _create_plot(
        results, var_names, time_range, t_start, t_duration, dt,
        batch_idx, model_name, figsize, title, filename
    )

    return fig, results


@beartype
def _extract_time_params_from_results(results: Dict[str, torch.Tensor],
                                      batch_idx: int) -> Tuple[float, float, float]:
    """Extract time parameters from simulation results - same format for both model types."""

    assert 't' in results, "Time not included in results, cannot extract it!"
    
    # Both return [batch_size, time] format
    time_tensor = results['t'][batch_idx, :]

    assert time_tensor.shape[0] > 1, "Time tensor is of unit length, can't extract dt"

    dt = float(time_tensor[1] - time_tensor[0])
    t_start = float(time_tensor[0])
    t_duration = float(time_tensor[-1] - time_tensor[0])
    
    return t_start, dt, t_duration


@beartype
def _select_variables(results: Dict[str, torch.Tensor],
                      variables: Optional[List[str]]) -> List[str]:
    """Select which variables to plot (time bookkeeping keys are never plotted)."""
    no_plot = ['t', 'dt', 't_start', 't_duration']

    if variables is None:
        var_names = list(results.keys())
    else:
        var_names = []
        for var in variables:
            assert var in results, f"{var} not in results for plotting"
            var_names.append(var)

    var_names = [v for v in var_names if v not in no_plot]
    assert var_names, "No variables found for plotting"
    return var_names


def _variable_unit(var: str) -> Tuple[str, bool]:
    """
    Infer the display unit (and whether it is a temperature) for a variable from
    the project naming convention (see developer.md). Temperatures are stored in
    Kelvin but displayed in degrees Celsius.

    Returns:
        (unit_label, is_temperature)
    """
    name = var.split('.')[-1]  # strip the "<node>." namespace prefix
    if 'T_' in name:                       # T_* -> temperature [K], shown as [°C]
        return '°C', True
    if 'Q_' in name or name.endswith('power') or 'heat_flow' in name:
        return 'W', False
    if 'P_' in name:
        return 'Pa', False
    if 'airflow' in name:
        return 'kg/s', False
    if 'position' in name:
        return '[0-1]', False
    if 'weather_factor' in name or 'integral_accumulator' in name:
        return '[-]', False
    return '', False


@beartype
def _create_plot(results: Dict[str, torch.Tensor],
                 var_names: List[str],
                 time_range: Optional[Tuple[float, float]],
                 t_start: float,
                 t_duration: float,
                 dt: float,
                 batch_idx: int,
                 model_name: str,
                 figsize: Optional[Tuple[float, float]],
                 title: Optional[str],
                 filename: Optional[str]) -> plt.Figure:
    """Create the actual matplotlib plot.

    Series are plotted against elapsed hours from the start of the run; the
    x-axis is formatted back to wall-clock time. ``time_range`` (start_hour,
    end_hour) zooms to a window of elapsed time.
    """
    zoomed = time_range is not None and (time_range[1] - time_range[0]) <= 4

    n_vars = len(var_names)
    if figsize is None:
        width = 12.0 if time_range else 8.0
        height = min(2.5 * n_vars, 20.0)  # Cap at reasonable height
        figsize = (width, height)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]

    # Absolute epoch time -> elapsed hours from the start of the run.
    time_sec = results['t'][batch_idx].detach().cpu().numpy().flatten()
    elapsed_hr = (time_sec - time_sec[0]) / 3600.0
    keep = None
    if time_range is not None:
        keep = (elapsed_hr >= time_range[0]) & (elapsed_hr <= time_range[1])

    for i, var in enumerate(var_names):
        data = results[var]
        if not hasattr(data, 'ndim') or data.ndim != 3:
            print(f"Skipping {var}: expected (batch, steps, features), "
                  f"got {getattr(data, 'shape', type(data))}")
            continue

        unit, is_temperature = _variable_unit(var)
        series = data[batch_idx].detach().cpu().numpy()  # [steps, features]
        if is_temperature:
            series = series - 273.15  # Kelvin -> Celsius for display only
        # State/output trajectories include the initial condition (length nsteps+1),
        # while inputs and 't' are length nsteps; align to the common length.
        n = min(len(elapsed_hr), series.shape[0])
        x = elapsed_hr[:n]
        y = series[:n]
        if keep is not None:
            mask = keep[:n]
            x, y = x[mask], y[mask]

        for feat in range(y.shape[1]):
            axes[i].plot(x, y[:, feat], label=f"{var} {feat}", linewidth=1.5)
        axes[i].set_ylabel(f"{var} [{unit}]" if unit else var)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best')

    # x-axis: format elapsed hours back to wall-clock time.
    time_formatter = _create_time_formatter(t_start, time_range)
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8 if zoomed else 6))
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=45 if zoomed else 0, ha='right' if zoomed else 'center')

    hour_of_day = floor((t_start % 86400) / 3600)
    hour_12 = ((hour_of_day - 1) % 12) + 1
    start_am_pm = "AM" if hour_of_day < 12 else "PM"
    axes[-1].set_xlabel(f"Time (starting {int(hour_12)} {start_am_pm})")

    if title is None:
        span_hours = (time_range[1] - time_range[0]) if time_range else (t_duration / 3600.0)
        title = (f"{model_name} Simulation: {span_hours:.1f} hours "
                 f"from {int(hour_12)} {start_am_pm} (dt={dt:.0f}s, batch={batch_idx})")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')

    return fig


@beartype
def _create_time_formatter(t_start: float,
                           time_range: Optional[Tuple[float, float]] = None):
    """Format an elapsed-hours x value as wall-clock time of day."""
    zoomed = time_range is not None and (time_range[1] - time_range[0]) <= 4

    def format_time_tick(x, pos):
        t_of_day = (t_start + x * 3600.0) % 86400  # x is elapsed hours
        h = int(t_of_day // 3600)
        h_12 = ((h - 1) % 12) + 1  # ((a-1) % p) + 1 maps 0 -> p
        m = int((t_of_day % 3600) // 60)
        am_pm = "AM" if h < 12 else "PM"
        return f"{h_12}:{m:02d} {am_pm}" if zoomed else f"{h_12} {am_pm}"

    return format_time_tick
