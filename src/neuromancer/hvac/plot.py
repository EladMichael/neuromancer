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

import os
if os.environ.get("RUNTIME_TYPING", 1):
    from beartype import beartype
else:
    # passthrough for no type checking
    def beartype(fn): return fn


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
        # Standard simulation parameters
        sim_kwargs = {
            't_duration': 86400.0,
            'dt': 300,
            't_start': 5.0*60*60,  # 5 AM
            **kwargs
        }

        # Both models use same method and return same format!
        results = model.simulate(**sim_kwargs)

        t_duration = sim_kwargs['t']
        dt = sim_kwargs['dt']
        t_start = sim_kwargs['t_start']
    else:
        # Extract time parameters from existing results
        t_start, dt, t_duration = _extract_time_params_from_results(results, batch_idx)

    # Get time vector for plotting - same for both model types
    # time_vec = results['t'][batch_idx, :]
    # time_vec = _create_time_vector(results, batch_idx, t_start, dt)

    # Select and filter variables
    var_names = _select_variables(results, variables)

    # Apply time range filtering - same logic for both
    # time_hours_plot, results_plot = _apply_time_range_filter(
    #     results, var_names, time_vec, time_range, t_start, batch_idx
    # )

    # Create the plot
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
def _create_time_vector(results: Dict[str, torch.Tensor],
                        batch_idx: int,
                        t_start: float,
                        dt: float) -> torch.Tensor:
    """Create time vector for plotting - same format for both model types."""

    # Both return [batch_size, time, ...] format
    # Find the maximum time dimension across all variables (excluding 't' and 'dt')
    data_vars = [k for k in results.keys() if k not in ['t', 'dt']]
    if data_vars:
        sample_result = results[data_vars[0]]
        n_steps = sample_result.shape[1]  # time is always dimension 1
    else:
        # Fallback to 't' if no other variables
        sample_result = next(iter(results.values()))
        n_steps = sample_result.shape[1]

    return torch.arange(t_start, t_start+dt*n_steps, dt)


@beartype
def _select_variables(results: Dict[str, torch.Tensor],
                      variables: Optional[List[str]]) -> List[str]:

    no_plot = ['t_start', 'dt', 't_duration']

    """Select which variables to plot."""
    if variables is None:
        # Plot all variables except time parameters
        var_names = list(results.keys())
    else:
        var_names = []
        for var in variables:
            assert var in results, f"{var} not in results for plotting"
            var_names.append(var)
            
    assert var_names, "No variables found for plotting"

    return [v for v in var_names if v not in no_plot]


# def _apply_time_range_filter(results: Dict[str, torch.Tensor],
#                              var_names: List[str],
#                              time_vec: torch.Tensor,
#                              time_range: Optional[Tuple[float, float]],
#                              t_start: float,
#                              batch_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Apply time range filtering to results - same format for both model types."""
#     results_plot = {}

#     # Process each variable and ensure time alignment
#     for var in var_names:
#         data = results[var][batch_idx, :].detach().cpu().numpy()

#         # Handle potential mismatch between time and data lengths
#         if len(data) != len(time_hours):
#             if len(data) == len(time_hours) + 1:
#                 # Data has one more point than time (initial condition + n steps)
#                 # Create extended time array
#                 dt_hours = time_hours[1] - time_hours[0] if len(time_hours) > 1 else 1/12  # 5min default
#                 extended_time = torch.cat([
#                     torch.tensor([time_hours[0] - dt_hours]),
#                     torch.tensor(time_hours)
#                 ])
#                 time_hours_for_var = extended_time
#             elif len(data) == len(time_hours) - 1:
#                 # Data has one fewer point than time
#                 time_hours_for_var = time_hours[:-1]
#             else:
#                 # Other mismatches - truncate to shorter length
#                 min_len = min(len(data), len(time_hours))
#                 data = data[:min_len]
#                 time_hours_for_var = time_hours[:min_len]
#         else:
#             time_hours_for_var = time_hours

#         # Store the aligned time for this variable
#         if var == var_names[0]:  # Use first variable's time as reference
#             time_hours_plot = time_hours_for_var

#         results_plot[var] = data

#     # Apply time range filtering if specified
#     if time_range is not None:
#         start_hour, end_hour = time_range
#         start_abs = t_start_hour + start_hour
#         end_abs = t_start_hour + end_hour

#         # Find indices for time range
#         mask = (time_hours_plot >= start_abs) & (time_hours_plot <= end_abs)
#         time_hours_plot = time_hours_plot[mask]

#         # Filter all results with the same mask
#         for var in var_names:
#             if len(results_plot[var]) == len(mask):
#                 results_plot[var] = results_plot[var][mask]
#             else:
#                 # Handle length mismatch in masking
#                 var_mask = mask[:len(results_plot[var])]
#                 results_plot[var] = results_plot[var][var_mask]

#     return time_hours_plot, results_plot


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
    """Create the actual matplotlib plot."""
    # Calculate figure size
    n_vars = len(var_names)
    if figsize is None:
        width = 12 if time_range else 8
        height = min(2.5 * n_vars, 20)  # Cap at reasonable height
        figsize = (width, height)

    # Create plots
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]

    time_vec = results['t'][batch_idx, :].cpu().numpy().flatten()

    for i, var in enumerate(var_names):
        data = results[var]

        if not hasattr(data, 'ndim'):
            print(f"Asked to plot {data} which seems to be scalar")
            continue
        elif data.ndim != 3:
            print(f"Data has wrong shape {data.shape}, should be (batch, steps, features)")

        # Everything should be shaped like (batches, steps, features)
        for feat in range(data.shape[2]):
            axes[i].plot(time_vec, data[batch_idx, :, feat], 
                         label=var+f' {feat}', linewidth=1.5)
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best')

        # Add value annotations for zoomed plots
        # if time_range and len(time_vec) < 100:
        #     # Annotate start and end values for short time series
        #     axes[i].annotate(f'{y_data[0]:.3f}',
        #                      xy=(time_vec[0], y_data[0]),
        #                      xytext=(5, 5), textcoords='offset points',
        #                      fontsize=8, alpha=0.7)
        #     axes[i].annotate(f'{y_data[-1]:.3f}',
        #                      xy=(time_vec[-1], y_data[-1]),
        #                      xytext=(5, 5), textcoords='offset points',
        #                      fontsize=8, alpha=0.7)

    # Apply time formatting
    time_formatter = _create_time_formatter(t_start, time_range)

    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

        if time_range and (time_range[1] - time_range[0]) <= 4:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Set xlabel
    if time_range and (time_range[1] - time_range[0]) <= 4:
        axes[-1].set_xlabel("Time")
    else:
        hour_of_day = floor((t_start % 86400) / 3600)
        hour_12 = ((hour_of_day - 1) % 12) + 1 
        start_am_pm = "AM" if hour_of_day < 12 else "PM"
        axes[-1].set_xlabel(f"Time (Starting {int(hour_12)} {start_am_pm})")

    # Set title
    if title is None:
        if time_range:
            title = (f"{model_name} Simulation: "
                     f"{time_range[1] - time_range[0]:.1f} hours "
                     f"(dt={dt:.1f}sec, batch={batch_idx})")
        else:
            hour_of_day = floor((t_start % 86400) / 3600)
            hour_12 = ((hour_of_day - 1) % 12) + 1 
            start_am_pm = "AM" if hour_of_day < 12 else "PM"
            title = (f"{model_name} Simulation: "
                     f"{t_duration/3600:.1f} hours from {int(hour_12)} {start_am_pm} "
                     f"(dt={dt:.1f}sec, batch={batch_idx})")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')

    return fig


@beartype
def _create_time_formatter(t_start: float,
                           time_range: Optional[Tuple[float, float]] = None):
    """Create time formatter for matplotlib."""

    def format_time_tick(x, pos):
        t_of_day = (t_start + x) % 86400
        # ((a-1) % p) + 1 makes the resulting zeros equal p
        h = int(t_of_day / 3600)
        h_12 = ((h-1) % 12) + 1  
        m = int(t_of_day / 60)
        am_pm = "AM" if h < 12 else "PM"

        if time_range and (time_range[1] - time_range[0]) <= 4:
            return f"{h_12}:{m:02d} {am_pm}"
        else:
            return f"{int(h_12)} {am_pm}"

    return format_time_tick
