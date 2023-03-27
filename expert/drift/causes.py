import typing
from datetime import timedelta
from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from expert.drift import __test_factory
from expert.drift.model import Drift, DriftCauses
from expert.drift.model import _Pair as Pair
from expert.utils import (
    compute_activity_arrival_rate,
    compute_activity_waiting_times,
    compute_resources_utilization_rate,
)


def __check_drift(
        reference_data: typing.Mapping[str, typing.Iterable[float]],
        running_data: typing.Mapping[str, typing.Iterable[float]],
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool],
) -> typing.Mapping[str, bool]:
    # Perform the statistical test over each pair of items in the given mappings to check if there has been a change
    return {
        key: test(reference_data[key], running_data[key]) if key in reference_data and key in running_data else True
        for key in set(list(reference_data.keys()) + list(running_data.keys()))
    }


def __plot_bars(ax: Axes,
                title: str,
                keys: list[str],
                reference: list[float],
                running: list[float],
                width: float = 0.3,
                space: float = 0.015) -> Axes:
    # Build the list of indices where the bars will be print
    ticks = np.arange(len(keys))
    # Put the bars for the reference dataset on their positions
    ax.bar(x=ticks, height=reference, width=width, color="royalblue", label="Reference")
    # Add the width plus the space between bars to offset the bar in the plot and prevent overlapping
    ax.bar(x=ticks + width + space, height=running, width=width, color="orangered", label="Running")
    ax.set_title(title)
    # Put the ticks in their position + (width + space) / 2 to center them below the bars, and add the labels rotated
    ax.set_xticks(ticks=ticks + (width + space) / 2, labels=keys, rotation=45, ha='right')
    ax.legend(loc="upper left")

    return ax


def explain_drift(
        drift_model: Drift,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool] = __test_factory(),
        granularity: timedelta = timedelta(hours=1),
) -> DriftCauses:
    """
    Find the actionable causes of a drift given the drift model.

    This function computes multiple metrics for the reference and running models and, based on that metrics, it provides
    insights about actionable causes for the drift, mainly related to the resources.


    Parameters
    ----------
    * `drift_model`:    *the model containing the running and reference events for the drift*
    * `test`:           *the test for evaluating if there are any difference between the reference and the running
                        models*

    Returns
    -------
    * the result of the drift detection with the drift causes
    """
    # Compute the different metrics
    reference_arrival_rate = compute_activity_arrival_rate(drift_model.reference_model, timeunit=granularity)
    running_arrival_rate = compute_activity_arrival_rate(drift_model.running_model, timeunit=granularity)

    reference_resource_utilization_rate = compute_resources_utilization_rate(drift_model.reference_model,
                                                                             timeunit=granularity)
    running_resource_utilization_rate = compute_resources_utilization_rate(drift_model.running_model,
                                                                           timeunit=granularity)

    reference_waiting_times = compute_activity_waiting_times(drift_model.reference_model)
    running_waiting_times = compute_activity_waiting_times(drift_model.running_model)

    # Build the result using the values previously computed
    return DriftCauses(
        model=Pair(
            reference=drift_model.reference_model,
            running=drift_model.running_model,
        ),
        case_duration=Pair(
            reference=drift_model.reference_durations,
            running=drift_model.running_durations,
        ),
        arrival_rate=Pair(
            reference=reference_arrival_rate,
            running=running_arrival_rate,
        ),
        resource_utilization_rate=Pair(
            reference=reference_resource_utilization_rate,
            running=running_resource_utilization_rate,
        ),
        waiting_time=Pair(
            reference=reference_waiting_times,
            running=running_waiting_times,
        ),
        arrival_rate_changed=any(
            __check_drift(
                reference_arrival_rate,
                running_arrival_rate,
                test=test,
            ).values(),
        ),
        resource_utilization_rate_changed=any(
            __check_drift(
                reference_resource_utilization_rate,
                running_resource_utilization_rate,
                test=test,
            ).values(),
        ),
        waiting_time_changed=any(
            __check_drift(
                {
                    key: [value.total_seconds() for value in reference_waiting_times[key]]
                    for key in reference_waiting_times
                },
                {
                    key: [value.total_seconds() for value in running_waiting_times[key]]
                    for key in running_waiting_times
                },
                test=test,
            ).values(),
        ),
    )

def plot_causes(causes: DriftCauses) -> Figure:
    """TODO"""
    fig, (ct, wt, ar, ru) = plt.subplots(nrows=4, ncols=1, layout="constrained", figsize=(8, 16))

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
    ct.legend(loc="upper left")



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
    fig.suptitle('Drift causes', size="xx-large")
    # Modify the figure layout to increase the gap between subplots
    fig.get_layout_engine().set(w_pad=8/72, h_pad=8/72)

    return fig

