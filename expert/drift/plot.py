"""This module contains the logic for plotting change causes"""

from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from expert.drift.model import DriftCauses


def __plot_bars(ax: Axes,
                title: str,
                keys: list[str],
                reference: list[float],
                running: list[float],
                width: float = 0.3,
                space: float = 0.0) -> Axes:
    # Build the list of indices where the bars will be print
    ticks = np.arange(len(keys))
    # Put the bars for the reference dataset on their positions
    ax.bar(x=ticks, height=reference, width=width, color="tab:blue", label="Reference")
    # Add the width plus the space between bars to offset the bar in the plot and prevent overlapping
    ax.bar(x=ticks + width + space, height=running, width=width, color="tab:red", label="Running")
    ax.set_title(title)
    # Put the ticks in their position + (width + space) / 2 to center them below the bars, and add the labels rotated
    ax.set_xticks(ticks=ticks + (width + space) / 2, labels=keys, rotation=45, ha="right")
    ax.legend(loc="upper left")

    return ax

def plot_causes(causes: DriftCauses) -> Figure:
    """Generate a plot with the before and after of the drift causes to easily visualize what changed"""
    fig, (ct, et, wt, ar, ru) = plt.subplots(nrows=5, ncols=1, layout="constrained", figsize=(8, 24))

    # Plot cycle times
    # Build the list of values for the cycle time and sort them
    reference = causes.case_duration.reference
    running = causes.case_duration.running

    reference_count=len(list(reference))
    running_count=len(list(running))
    max_x = max(reference_count, running_count)

    ref_x = np.linspace(start=0, stop=max_x, num=reference_count)
    run_x = np.linspace(start=0, stop=max_x, num=running_count)

    ct.plot(ref_x, reference, color="tab:blue", label="Reference")
    ct.plot(run_x, running, color="tab:red", label="Running")
    ct.set_title("Process cycle time")
    ct.set_ylabel("time (s)")
    ct.set_xticks([])
    ct.set_xbound(lower=-0.5, upper=max_x+0.5)
    ct.legend(loc="upper left")

    # Plot execution times per activity
    # Get the set of keys (they can be different in reference and running, so we concat and build a set)
    keys = sorted(set(list(causes.execution_time.reference.keys()) + list(causes.execution_time.running.keys())))
    # Compute the mean for each key, and assign 0.0 if the activity was not present
    reference = [
        mean([value.total_seconds() for value in causes.execution_time.reference[key]])
        if key in causes.execution_time.reference else 0.0 for key in keys
    ]
    running = [
        mean([value.total_seconds() for value in causes.execution_time.running[key]])
        if key in causes.execution_time.running else 0.0 for key in keys
    ]
    __plot_bars(ax=et, title="execution time/activity", reference=reference, running=running, keys=keys)

    # Plot waiting times per activity
    # Get the set of keys (they can be different in reference and running, so we concat and build a set)
    keys = sorted(set(list(causes.waiting_time.reference.keys()) + list(causes.waiting_time.running.keys())))
    # Compute the mean for each key, and assign 0.0 if the activity was not present
    reference = [
        mean([value.total_seconds() for value in causes.waiting_time.reference[key]])
        if key in causes.waiting_time.reference else 0.0 for key in keys
    ]
    running = [
        mean([value.total_seconds() for value in causes.waiting_time.running[key]])
        if key in causes.waiting_time.running else 0.0 for key in keys
    ]
    __plot_bars(ax= wt, title ="waiting time/activity", reference=reference, running=running, keys=keys)

    # Plot arrival rate per activity
    keys = sorted(set(list(causes.arrival_rate.reference.keys()) + list(causes.arrival_rate.running.keys())))
    reference = [
        mean(causes.arrival_rate.reference[key]) if key in causes.arrival_rate.reference else 0.0 for key in keys
    ]
    running = [
        mean(causes.arrival_rate.running[key]) if key in causes.arrival_rate.running else 0.0 for key in keys
    ]
    __plot_bars(ax=ar, title="arrival rate/activity", reference=reference, running=running, keys=keys)

    # Plot resource utilization rate
    keys = sorted(
        set(
            list(causes.resource_utilization_rate.reference.keys()) +
            list(causes.resource_utilization_rate.running.keys()),
            ),
    )
    reference = [
        mean(causes.resource_utilization_rate.reference[key])
        if key in causes.resource_utilization_rate.reference else 0.0 for key in keys
    ]
    running = [
        mean(causes.resource_utilization_rate.running[key])
        if key in causes.resource_utilization_rate.running else 0.0 for key in keys
    ]
    __plot_bars(ax=ru, title="utilization rate/resource", reference=reference, running=running, keys=keys)

    # Set the figure title
    fig.suptitle("Drift causes", size="xx-large")
    # Modify the figure layout to increase the gap between subplots
    fig.get_layout_engine().set(w_pad=8/72, h_pad=8/72)

    return fig
