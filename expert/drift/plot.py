"""This module contains the logic for plotting change causes"""
import math
from datetime import timedelta
from statistics import mean

import numpy as np
from anytree import Node, RenderTree
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from expert.drift.causality import CAUSE_DETAILS_TYPE
from expert.drift.features import DriftFeatures
from expert.logger import LOGGER


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

def plot_features(causes: DriftFeatures) -> Figure:
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


def print_causes(drift_causes: Node) -> None:
    """Pretty-print the drift causes"""
    LOGGER.notice("drift causes:")
    for pre, fill, node in RenderTree(drift_causes):
        match node.type:
            case CAUSE_DETAILS_TYPE.SUMMARY_PAIR:
                factor: float = (
                    node.details.running.mean / node.details.reference.mean
                    if node.details.running.mean != 0.0 and node.details.reference.mean != 0
                    else math.inf
                )

                LOGGER.notice("    %s%s x%.2f (from %s to %s)",
                              pre, node.name.lower(), factor,
                              timedelta(seconds=node.details.reference.mean),
                              timedelta(seconds=node.details.running.mean))
            case CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY:
                LOGGER.notice("    %s%s for %d activities", pre, node.name.lower(), len(node.details))

                for (activity, details) in node.details.items():
                    factor: float = (
                        details.running.mean/details.reference.mean
                        if details.running.mean != 0.0 and details.reference.mean != 0
                        else math.inf
                    )

                    LOGGER.info(
                        "    %s    '%s' %s x%.2f (from %s to %s)",
                        fill,
                        activity,
                        node.name.lower(),
                        factor,
                        timedelta(seconds=details.reference.mean)
                        if details.unit == "duration"
                        else f"{details.reference.mean} {details.unit}",
                        timedelta(seconds=details.running.mean)
                        if details.unit == "duration"
                        else f"{details.reference.mean} {details.unit}",
                    )
            case CAUSE_DETAILS_TYPE.DIFFERENCE_PER_ACTIVITY:
                LOGGER.notice("    %s%s for %d activities", pre, node.name.lower(), len(node.details))

                for (activity, details) in node.details.items():
                    LOGGER.info("    %s    '%s' %s %s", fill, activity, node.name.lower(), ", ".join(details))
            case CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE:
                LOGGER.notice("    %s%s for %d activities", pre, node.name.lower(), len(node.details))

                for (activity, resources) in node.details.items():
                    LOGGER.info(
                        "    %s    '%s' %s",
                        fill,
                        activity,
                        node.name.lower(),
                    )

                    for (resource, summary) in resources.items():
                        LOGGER.info(
                            "    %s        '%s' mean time is %s (reference mean time was %s)",
                            fill,
                            resource,
                            timedelta(seconds=summary.running.mean),
                            timedelta(seconds=summary.reference.mean),
                        )

            case _:
                LOGGER.notice("    %s%s", pre, node.name.lower())
                LOGGER.info("    %s    %s", fill, node.details)

