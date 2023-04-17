"""This module contains the logic for evaluating the actionable causes of a drift in the cycle time of a process"""
import typing
from dataclasses import dataclass
from datetime import timedelta

from lazy import lazy

from expert.drift.model import Drift
from expert.utils.activities import (
    compute_activity_arrival_rate,
    compute_activity_batch_sizing,
    compute_activity_batching_times,
    compute_activity_contention_times,
    compute_activity_execution_times,
    compute_activity_prioritization_times,
    compute_activity_resources,
    compute_activity_waiting_times,
)
from expert.utils.resources import compute_resources_availability_calendars, compute_resources_utilization_rate

_T = typing.TypeVar("_T")


@dataclass
class Pair(typing.Generic[_T]):
    """A pair of measurement values, used as reference and running values for drift detection"""

    reference: _T
    """The reference values"""
    running: _T
    """The running values"""
    unit: str = ""
    """The unit of measurement"""


class DriftFeatures:
    """
    The features that describe the reference and running models for a given change.

    When created, this object computes multiple metrics for the reference and running models as the execution times, the
    arrival rates, the resources utilization rates or the waiting times, both aggregated and decomposed in its different
    components.


    Parameters
    ----------
    * `drift_model`:    *the model containing the running and reference events for the drift*
    * `granularity`:    *the granularity used for computing time-binned features*

    Returns
    -------
    * the result of the drift detection with the drift causes
    """

    model: Drift
    """The drift model"""
    granularity: timedelta
    """Tha granularity for computing the features that are binned by time"""

    def __init__(self: typing.Self, model: Drift, granularity: timedelta = timedelta(hours=1)) -> None:
        self.model = model
        self.granularity = granularity

    @lazy
    def case_duration(self: typing.Self) -> Pair[typing.Iterable[float]]:
        """Get the case durations for the running and reference models"""
        return Pair(
            reference=self.model.reference_durations,
            running=self.model.running_durations,
            unit="duration",
        )
    @lazy
    def execution_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[timedelta]]]:
        """Get the set of execution times for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_execution_times(self.model.reference_model),
            running=compute_activity_execution_times(self.model.running_model),
            unit="duration",
        )
    @lazy
    def arrival_rate(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[float]]]:
        """Get the arrival rate for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_arrival_rate(self.model.reference_model, self.granularity),
            running=compute_activity_arrival_rate(self.model.running_model, self.granularity),
            unit=f"instances per {self.granularity}",
        )
    @lazy
    def resource_utilization_rate(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[float]]]:
        """Get the utilization rate for each resource for the running and the reference models"""
        return Pair(
            reference=compute_resources_utilization_rate(self.model.reference_model, self.granularity),
            running=compute_resources_utilization_rate(self.model.running_model, self.granularity),
            unit=f"instances per {self.granularity}",
        )
    @lazy
    def waiting_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[timedelta]]]:
        """Get the set of waiting times for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_waiting_times(self.model.reference_model),
            running=compute_activity_waiting_times(self.model.running_model),
            unit="duration",
        )
    @lazy
    def batching_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to batching for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_batching_times(self.model.reference_model),
            running=compute_activity_batching_times(self.model.running_model),
            unit="duration",
        )
    @lazy
    def batch_sizes(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """Get the batch sizes for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_batch_sizing(self.model.reference_model),
            running=compute_activity_batch_sizing(self.model.running_model),
            unit="instances",
        )
    @lazy
    def contention_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to contention for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_contention_times(self.model.reference_model),
            running=compute_activity_contention_times(self.model.running_model),
            unit="duration",
        )
    @lazy
    def prioritization_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to prioritization for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_prioritization_times(self.model.reference_model),
            running=compute_activity_prioritization_times(self.model.running_model),
            unit="duration",
        )
    @lazy
    def resources_allocation(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[timedelta]]]:
        """Get the resources allocation for each activity for the running and the reference models"""
        return Pair(
            reference=compute_activity_resources(self.model.reference_model),
            running=compute_activity_resources(self.model.running_model),
            unit="resources per activity",
        )
    @lazy
    def resources_availability(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[bool]]]:
        """Get resources availability calendars for each resource for the running and the reference models"""
        return Pair(
            reference=compute_resources_availability_calendars(self.model.reference_model, granularity=self.granularity),
            running=compute_resources_availability_calendars(self.model.running_model, granularity=self.granularity),
            unit="resources calendars",
        )
