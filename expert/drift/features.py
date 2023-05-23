"""This module contains the logic for evaluating the actionable causes of a drift in the cycle time of a process"""
import typing
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from expert.drift.model import Drift
from expert.model import Activity, Event, Log
from expert.utils.waiting import decompose_waiting_times

_T = typing.TypeVar("_T")
_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")

def _aggregate(
        log: Log,
        *,
        key_extractor: typing.Callable[[Event], _K],
        value_extractor: typing.Callable[[Event], _V],
) -> typing.Mapping[_K, typing.Iterable[_V]]:
    # group events by key and store the value for each event
    grouped = defaultdict(list)
    for event in log:
        grouped[key_extractor(event)].append(value_extractor(event))

    return grouped


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
        self.__reference_waiting_times = decompose_waiting_times(model.reference_model)
        self.__running_waiting_times = decompose_waiting_times(model.running_model)


    @property
    def case_duration(self: typing.Self) -> Pair[typing.Iterable[float]]:
        """Get the case durations for the running and reference models"""
        return Pair(
            reference=self.model.reference_durations,
            running=self.model.running_durations,
            unit="duration",
        )


    @property
    def execution_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the set of execution times for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.execution_time,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.execution_time,
            ),
            unit="duration",
        )


    @property
    def waiting_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the set of waiting times for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.total.duration,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.total.duration,
            ),
            unit="duration",
        )


    @property
    def batching_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to batching for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.batching.duration,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.batching.duration,
            ),
            unit="duration",
        )


    @property
    def contention_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to contention for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.contention.duration,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.contention.duration,
            ),
            unit="duration",
        )


    @property
    def prioritization_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to prioritization for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.prioritization.duration,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.prioritization.duration,
            ),
            unit="duration",
        )


    @property
    def availability_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to resources unavailability for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.availability.duration,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.availability.duration,
            ),
            unit="duration",
        )


    @property
    def extraneous_time(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[timedelta]]]:
        """Get the part of waiting times due to extraneous factors for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.extraneous.duration,
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: event.waiting_time.extraneous.duration,
            ),
            unit="duration",
        )



    # # REFACTOR FROM HERE DOWN
    # @lazy
    # def arrival_rate(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[float]]]:
    #     """Get the arrival rate for each activity for the running and the reference models"""
    #     return Pair(
    #         reference=compute_activity_arrival_rate(self.model.reference_model, self.granularity),
    #         running=compute_activity_arrival_rate(self.model.running_model, self.granularity),
    #         unit=f"instances per {self.granularity}",
    #     )
    #
    #
    # @lazy
    # def resource_utilization_rate(self: typing.Self) -> Pair[typing.Mapping[Resource, typing.Iterable[float]]]:
    #     """Get the utilization rate for each resource for the running and the reference models"""
    #     return Pair(
    #         reference=compute_resources_utilization_rate(self.model.reference_model, self.granularity),
    #         running=compute_resources_utilization_rate(self.model.running_model, self.granularity),
    #         unit=f"instances per {self.granularity}",
    #     )
    #
    #
    # @lazy
    # def resources_allocation(self: typing.Self) -> Pair[typing.Mapping[Activity, typing.Iterable[Resource]]]:
    #     """Get the resources allocation for each activity for the running and the reference models"""
    #     return Pair(
    #         reference=compute_activity_resources(self.model.reference_model),
    #         running=compute_activity_resources(self.model.running_model),
    #         unit="resources per activity",
    #     )
    #
    #
    # @lazy
    # def resources_availability(self: typing.Self) -> Pair[typing.Mapping[Resource, typing.Iterable[bool]]]:
    #     """Get resources availability calendars for each resource for the running and the reference models"""
    #     return Pair(
    #         reference=compute_resources_availability_calendars(
    #             self.model.reference_model,
    #             granularity=self.granularity,
    #         ),
    #         running=compute_resources_availability_calendars(
    #             self.model.running_model,
    #             granularity=self.granularity,
    #         ),
    #         unit="resources calendars",
    #     )
    #
    #
    # @lazy
    # def batch_sizes(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
    #     """Get the batch sizes for each activity for the running and the reference models"""
    #     return Pair(
    #         reference=compute_activity_batch_sizing(self.model.reference_model),
    #         running=compute_activity_batch_sizing(self.model.running_model),
    #         unit="instances",
    #     )
