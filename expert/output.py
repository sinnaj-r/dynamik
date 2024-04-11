"""This module contains the logic for plotting change causes"""
import datetime
import json
import typing
from statistics import mean

import numpy as np
import yaml
from anytree import AnyNode, RenderTree
from intervaltree import Interval, IntervalTree
from matplotlib.axes import Axes
from sortedcontainers.sorteddict import SortedDict

from expert.process_model import Event
from expert.utils.logger import LOGGER
from expert.utils.model import Pair
from expert.utils.rules import ConfusionMatrix


def __plot_bars(axes: Axes,
                reference: typing.Mapping[str, typing.Iterable[int]],
                running: typing.Mapping[str, typing.Iterable[int]],
                *,
                aggregation: typing.Callable[[typing.Iterable[int]], float] = mean,
                title: str,
                width: float = 0.3,
                space: float = 0.0) -> Axes:
    # Build the list of indices where the bars will be print
    keys = sorted(set(list(reference.keys()) + list(running.keys())))
    ticks = np.arange(len(keys))
    # Put the bars for the reference dataset on their positions
    sorted_data = SortedDict(reference)

    axes.bar(
        x=ticks,
        height=[aggregation(values) for values in sorted_data.values()],
        width=width,
        color="tab:blue",
        label="Reference",
    )

    sorted_data = SortedDict(running)

    # Add the width plus the space between bars to offset the bar in the plot and prevent overlapping
    axes.bar(
        x=ticks + width + space,
        height=[aggregation(values) for values in sorted_data.values()],
        width=width,
        color="tab:red",
        label="Running",
    )
    # Set the plot title
    axes.set_title(title)
    # Put the ticks in their position + (width + space) / 2 to center them below the bars, and add the labels rotated
    axes.set_xticks(ticks=ticks + (width + space) / 2, labels=keys, rotation=45, ha="right")
    axes.legend(loc="upper left")
    axes.set_ylim(ymin=0.0)

    return axes


def __plot_continuous(axes: Axes,
                      reference: typing.Iterable[int],
                      running: typing.Iterable[int],
                      *,
                      title: str) -> Axes:
    reference_count = len(list(reference))
    running_count = len(list(running))
    max_x = max(reference_count, running_count)

    ref_x = np.linspace(start=0, stop=max_x, num=reference_count)
    run_x = np.linspace(start=0, stop=max_x, num=running_count)

    axes.plot(ref_x, list(reference), color="tab:blue", label="Reference")
    axes.plot(run_x, list(running), color="tab:red", label="Running")
    axes.set_title(title, pad=10)
    axes.set_ylabel("time (s)")
    axes.set_xticks([])
    axes.set_xbound(lower=-0.5, upper=max_x + 0.5)
    axes.set_xlabel("\n")
    axes.legend(loc="upper left")
    axes.set_ylim(ymin=0.0)

    return axes


# def plot_case_features(features: TimesPerCase) -> Figure:
#     """Generate a plot with the before and after of the drift causes to easily visualize what changed"""
#     fig, (ct, pt, ept, ipt, wt, ruwt, rcwt, pwt, bwt, ewt) = plt.subplots(
#         nrows=10,
#         ncols=1,
#         layout="constrained",
#         sharex="all",
#         figsize=(20, 40),
#     )
#
#     # Plot cycle times
#     __plot_continuous(ct, features.cycle_time.reference, features.cycle_time.running, title="Cycle time")
#
#     # Plot processing times
#     __plot_continuous(pt, features.processing_time.reference, features.processing_time.running, title="Processing time")
#
#     # Plot effective processing times
#     __plot_continuous(ept, features.effective_time.reference, features.effective_time.running, title="Effective processing time")
#
#     # Plot idle processing times
#     __plot_continuous(ipt, features.idle_time.reference, features.idle_time.running, title="Idle processing time")
#
#     # Plot waiting times
#     __plot_continuous(wt, features.waiting_time.reference, features.waiting_time.running, title="Waiting time")
#
#     # Plot waiting times due to resource unavailability
#     __plot_continuous(ruwt, features.availability_time.reference, features.availability_time.running, title="Waiting time (resource unavailability)")
#
#     # Plot waiting times due to resource contention
#     __plot_continuous(rcwt, features.contention_time.reference, features.contention_time.running, title="Waiting time (contention)")
#
#     # Plot waiting times due to prioritization
#     __plot_continuous(pwt, features.prioritization_time.reference, features.prioritization_time.running, title="Waiting time (prioritization)")
#
#     # Plot waiting times due to batching
#     __plot_continuous(bwt, features.batching_time.reference, features.batching_time.running, title="Waiting time (batching)")
#
#     # Plot waiting times due to exogenous factors
#     __plot_continuous(ewt, features.extraneous_time.reference, features.extraneous_time.running, title="Waiting time (extraneous)")
#
#     return fig
#
#
# def plot_activity_features(features: TimesPerActivity) -> Figure:
#     """Generate a plot with the features at an activity level"""
#     # Generate a plot with the before and after of the drift causes to easily visualize what changed
#     fig, (pt, ept, ipt, wt, ruwt, rcwt, pwt, bwt, ewt) = plt.subplots(
#         nrows=9,
#         ncols=1,
#         layout="constrained",
#         figsize=(20, 40),
#     )
#
#     __plot_bars(pt, features.processing_time.reference, features.processing_time.running, title="Processing time (avg)")
#
#     # Plot effective processing times
#     __plot_bars(ept, features.effective_time.reference, features.effective_time.running, title="Effective processing time (avg)")
#
#     # Plot idle processing times
#     __plot_bars(ipt, features.idle_time.reference, features.idle_time.running, title="Idle processing time (avg)")
#
#     # Plot waiting times
#     __plot_bars(wt, features.waiting_time.reference, features.waiting_time.running, title="Waiting time (avg)")
#
#     # Plot waiting times due to resource unavailability
#     __plot_bars(ruwt, features.availability_time.reference, features.availability_time.running, title="Waiting time (resource unavailability) (avg)")
#
#     # Plot waiting times due to resource contention
#     __plot_bars(rcwt, features.contention_time.reference, features.contention_time.running, title="Waiting time (contention) (avg)")
#
#     # Plot waiting times due to prioritization
#     __plot_bars(pwt, features.prioritization_time.reference, features.prioritization_time.running, title="Waiting time (prioritization) (avg)")
#
#     # Plot waiting times due to batching
#     __plot_bars(bwt, features.batching_time.reference, features.batching_time.running, title="Waiting time (batching) (avg)")
#
#     # Plot waiting times due to exogenous factors
#     __plot_bars(ewt, features.extraneous_time.reference, features.extraneous_time.running, title="Waiting time (extraneous) (avg)")
#
#     return fig


def print_causes(drift_causes: AnyNode) -> None:
    """Pretty-print the drift causes"""
    LOGGER.notice("drift causes:")
    tree = RenderTree(drift_causes).by_attr('what')
    for line in tree.splitlines():
        LOGGER.notice(line)

    # plt.hist([value.total_seconds() for value in drift_causes.data.reference])
    # plt.title("Reference waiting time histogram")
    # plt.show()
    #
    # plt.hist([value.total_seconds() for value in drift_causes.data.running])
    # plt.title("Running waiting time histogram")
    # plt.show()
    # histogram = plt.hist([value.total_seconds() for value in drift_causes.data.reference])


def export_causes(drift_causes: AnyNode, *, excluded_fields: typing.Iterable[str] = ("data",), filename: str | None = None) -> dict:
    """Export the current causes tree to a YAML representation"""
    LOGGER.notice("exporting full causes tree")
    # transform the tree to a dict and return it
    tree = __to_dict(drift_causes, excluded_fields)

    if filename is not None:
        extension = filename.split(".")[-1]
        with open(filename, mode="w") as file:
            match extension:
                case "json":
                    json.dump(tree, file)
                case "yaml":
                    yaml.dump(tree, file, default_flow_style=False)

    return tree


def __to_dict(data: typing.Any, excluded_fields: typing.Iterable[str]) -> typing.Any:
    if isinstance(data, dict):
        return {key: __to_dict(value, excluded_fields) for key, value in data.items() if key not in excluded_fields}

    if isinstance(data, AnyNode):
        node = {}
        if "what" not in excluded_fields:
            node["what"] = data.what

        if "how" not in excluded_fields:
            node["how"] = __to_dict(data.how, excluded_fields)

        if "data" not in excluded_fields:
            node["data"] = __to_dict(data.data, excluded_fields)

        if not data.is_leaf and "causes" not in excluded_fields:
            node["causes"] = [__to_dict(child, excluded_fields) for child in data.children]

        if "changes_per_activity" in data.__dict__ and "changes_per_activity" not in excluded_fields:
            node["changes_per_activity"] = __to_dict(data.changes_per_activity, excluded_fields)

        return node

    if isinstance(data, Event):
        return __to_dict(data.asdict(fields=("case", "activity", "resource", "attributes", "start", "end", "enabled")), excluded_fields)

    if isinstance(data, ConfusionMatrix):
        return __to_dict(data.__dict__, excluded_fields)

    if isinstance(data, Pair):
        return {
            "reference": __to_dict(data.reference, excluded_fields),
            "running": __to_dict(data.running, excluded_fields),
            "unit": data.unit,
        }

    if type(data).__name__ == "DescribeResult":
        return {
            "observations": int(data.nobs),
            "min": float(data.minmax[0]),
            "max": float(data.minmax[-1]),
            "mean": float(data.mean),
            "variance": float(data.variance),
            "skewness": float(data.skewness) if not np.isnan(data.skewness) else None,
            "kurtosis": float(data.kurtosis) if not np.isnan(data.kurtosis) else None,
        }

    if isinstance(data, Interval):
        return {
            "begin": __to_dict(data.begin, excluded_fields),
            "end": __to_dict(data.end, excluded_fields),
        }

    if isinstance(data, list | tuple | set | IntervalTree):
        return [
            __to_dict(value, excluded_fields) for value in data
        ]

    if isinstance(data, datetime.datetime):
        return data.isoformat()

    if isinstance(data, datetime.timedelta):
        return data.__str__()

    return data

