"""This module contains the definition of the model used for drift detection in the cycle time of a process."""

from __future__ import annotations

import enum
import typing
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property

import scipy
from anytree import AnyNode
from sortedcontainers import SortedSet

from expert.logger import LOGGER
from expert.model import Event, Log
from expert.utils.batching import compute_batches
from expert.utils.processing import decompose_processing_times
from expert.utils.statistical_tests import continuous_test
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


DriftCauses: typing.TypeAlias = AnyNode | None


@dataclass
class Pair(typing.Generic[_T]):
    """A pair of measurement values, used as reference and running values for drift detection"""

    reference: _T | None
    """The reference values"""
    running: _T | None
    """The running values"""
    unit: str = ""
    """The unit of measurement"""


@dataclass
class Drift:
    """The drift, with its level and the data that lead to the detection"""

    level: DriftLevel
    reference_model: typing.Iterable[Event] | None = None
    running_model: typing.Iterable[Event] | None = None
    reference_durations: typing.Iterable[float] | None = None
    running_durations: typing.Iterable[float] | None = None

    def __post_init__(self: typing.Self) -> None:
        # if the drift has been confirmed, compute the features
        if self.level == DriftLevel.CONFIRMED:
            # compute the batches
            compute_batches(self.reference_model)
            compute_batches(self.running_model)
            # decompose processing times
            decompose_processing_times(self.reference_model)
            decompose_processing_times(self.running_model)
            # decompose waiting times
            decompose_waiting_times(self.reference_model)
            decompose_waiting_times(self.running_model)

    @cached_property
    def case_features(self: typing.Self) -> TimesPerCase:
        """TODO docs"""
        return TimesPerCase(self)

    @cached_property
    def activity_features(self: typing.Self) -> TimesPerActivity:
        """TODO docs"""
        return TimesPerActivity(self)

    @cached_property
    def activities(self: typing.Self) -> typing.Iterable[str]:
        """TODO DOCS"""
        return SortedSet(event.activity for event in (list(self.reference_model) + list(self.running_model)) if
                         event.activity is not None)

    @cached_property
    def resources(self: typing.Self) -> typing.Iterable[str]:
        """TODO DOCS"""
        return SortedSet(event.resource for event in (list(self.reference_model) + list(self.running_model)) if
                         event.resource is not None)


class DriftLevel(enum.Enum):
    """The drift level. Can be no drift, drift warning or confirmed drift."""

    NONE = 0
    WARNING = 1
    CONFIRMED = 2


NO_DRIFT: Drift = Drift(level=DriftLevel.NONE)


class _Model:
    # The date and time when the model starts
    start: datetime | None = None
    # The date and time when the model ends
    end: datetime | None = None
    # The collection of events used as the model
    data: typing.MutableSequence[Event] = []

    @property
    def empty(self: typing.Self) -> bool:
        return len(self.data) == 0

    def prune(self: typing.Self) -> None:
        LOGGER.debug("pruning model")
        # Remove all events that are out of the overlapping region from the running model
        self.data = [event for event in self.data if event.start > self.start and event.end < self.end]

    def add(self: typing.Self, event: Event) -> None:
        self.data.append(event)


class DriftModel:
    """Stores the model that will be used to detect drifts in the process."""

    # The size of the reference and running models, in time units
    __timeframe_size: timedelta
    # The number of drift warnings to confirm a drift
    __warnings_to_confirm: int = 0
    # The period considered as a warm-up
    __warm_up: timedelta
    # The overlap between running models
    __overlap: timedelta = timedelta()
    # The reference model
    __reference_model: _Model = _Model()
    # The running model
    __running_model: _Model = _Model()
    # The collection of detection results
    __drifts: typing.MutableSequence[Drift] = deque([NO_DRIFT], maxlen=1)

    def __init__(
            self: typing.Self,
            *,
            timeframe_size: timedelta,
            warm_up: timedelta = timedelta(),
            overlap_between_models: timedelta = timedelta(),
            warnings_to_confirm: int = 3,
    ) -> None:
        """
        Create a new empty drift detection model with the given timeframe size and limit activities.

        Parameters
        ----------
        * `timeframe_size`:         *the timeframe used to build the reference and running models*
        * `warm_up`:                *the warm-up period during which events will be discarded*
        * `overlap_between_models`: *the overlapping between running models (must be smaller than the timeframe size)*
        * `warnings_to_confirm`:    *the number of consecutive detections needed for confirming a drift*
        """
        self.__timeframe_size = timeframe_size
        self.__warm_up = warm_up
        self.__warnings_to_confirm = warnings_to_confirm
        self.__drifts = deque([NO_DRIFT] * warnings_to_confirm, maxlen=warnings_to_confirm)
        self.__overlap = overlap_between_models

    @property
    def reference_model(self: typing.Self) -> typing.Iterable[Event]:
        """The list of events that are used as a reference model against which changes are checked"""
        return tuple(self.__reference_model.data)

    @property
    def running_model(self: typing.Self) -> typing.Iterable[Event]:
        """The list of events used as the running model being checked for changes"""
        return tuple(self.__running_model.data)

    @property
    def drift(self: typing.Self) -> Drift:
        """The last iteration drift status"""
        return self.__drifts[-1]

    def __update_reference_model(self: typing.Self, event: Event) -> None:
        LOGGER.debug("updating reference model")
        # Append the event to the list of events in the reference model
        self.__reference_model.add(event)

    def __update_running_model(self: typing.Self, event: Event) -> None:
        LOGGER.debug("updating running model")
        # Update events in the running model
        self.__running_model.add(event)

    def __update_drifts(self: typing.Self) -> None:
        LOGGER.debug("updating drifts")

        if continuous_test(
                [event.cycle_time.total_seconds() for event in self.__reference_model.data],
                [event.cycle_time.total_seconds() for event in self.__running_model.data],
        ):
            self.__drifts.append(
                Drift(
                    level=DriftLevel.WARNING,
                    reference_model=tuple(self.reference_model),
                    running_model=tuple(self.running_model),
                ),
            )
            LOGGER.verbose(
                "drift warning in the cycle time between reference timeframe (%s - %s) and running timeframe (%s - %s)",
                self.__reference_model.start, self.__reference_model.end,
                self.__running_model.start, self.__running_model.end)
            # If all the detections are warnings, replace by a confirmation
            if all(drift.level == DriftLevel.WARNING for drift in self.__drifts):
                LOGGER.verbose("drift confirmed between reference timeframe (%s - %s) and running timeframe (%s - %s)",
                               self.__reference_model.start, self.__reference_model.end,
                               self.__running_model.start, self.__running_model.end)
                self.__drifts.pop()
                self.__drifts.append(Drift(
                    level=DriftLevel.CONFIRMED,
                    reference_model=self.reference_model,
                    running_model=self.running_model,
                ))
        # If the model does not present a drift add a NONE
        else:
            LOGGER.verbose("no drift between reference timeframe (%s - %s) and running timeframe (%s, %s)",
                           self.__reference_model.start, self.__reference_model.end,
                           self.__running_model.start, self.__running_model.end)
            self.__drifts.append(NO_DRIFT)

    def reset(self: typing.Self) -> None:
        """Reset the models to their initial state"""
        self.__reference_model = _Model()
        self.__running_model = _Model()
        self.__drifts = deque([NO_DRIFT] * self.__warnings_to_confirm, maxlen=self.__warnings_to_confirm)

    def update(self: typing.Self, event: Event) -> None:
        """
        Update the model with a new event and check if it presents a drift.

        If the event lies in the reference time window it will be added to both the reference and the running model.
        Otherwise, only the running model will be updated.

        Parameters
        ----------
        * `event`: *the new event to be added to the model*
        """
        # Update the reference model timeframe if it is not initialized
        if self.__reference_model.start is None:
            # Set the reference model start and end dates
            self.__reference_model.start = event.start + self.__warm_up
            self.__reference_model.end = self.__reference_model.start + self.__timeframe_size
            LOGGER.debug("updating reference model to timeframe (%s - %s)",
                         self.__reference_model.start, self.__reference_model.end)
        # Update the running model timeframe if it is not initialized
        if self.__running_model.start is None:
            # Set the running model start and end
            self.__running_model.start = self.__reference_model.end - self.__overlap
            self.__running_model.end = self.__running_model.start + self.__timeframe_size
            LOGGER.debug("updating running model to timeframe (%s - %s)",
                         self.__running_model.start, self.__running_model.end)
        # Drop the event if it is part of the warm-up period
        if event.start < self.__reference_model.start:
            LOGGER.spam("dropping warm-up event %r", event)
        # If the event is part of the reference model, update it
        if self.__reference_model.start <= event.start <= event.end <= self.__reference_model.end:
            LOGGER.debug("updating reference model (timeframe %s - %s) with event %r", self.__reference_model.start,
                         self.__reference_model.end, event)
            self.__update_reference_model(event)
        # If the event is part of the running model, update it
        if self.__running_model.start <= event.start <= event.end <= self.__running_model.end:
            LOGGER.spam("updating running model (timeframe %s - %s) with event %r", self.__reference_model.start,
                        self.__reference_model.end, event)
            self.__update_running_model(event)
        # If the event is out of the running model, update the limits and prune the running model content
        if event.end > self.__running_model.end:
            if self.__running_model.empty:
                LOGGER.warning("no event executed in the timeframe %s - %s",
                               self.__running_model.start, self.__running_model.end)
            else:
                # Update the drifts
                LOGGER.verbose(
                    "checking drift between reference timeframe %s - %s (%s) and running timeframe %s - %s (%s)",
                    self.__reference_model.start, self.__reference_model.end,
                    scipy.stats.describe([event.cycle_time.total_seconds() for event in self.__reference_model.data]),
                    self.__running_model.start, self.__running_model.end,
                    scipy.stats.describe([event.cycle_time.total_seconds() for event in self.__running_model.data]))
                self.__update_drifts()

            # Update the running model limits until the event lies in the timeframe
            while event.end > self.__running_model.end:
                self.__running_model.start = self.__running_model.end - self.__overlap
                self.__running_model.end = self.__running_model.start + self.__timeframe_size
                LOGGER.debug("updating running model to timeframe (%s - %s)",
                             self.__running_model.start, self.__running_model.end)

                # Delete outdated events from the running model
                LOGGER.debug("pruning running model (timeframe %s - %s)",
                             self.__running_model.start, self.__running_model.end)
                self.__running_model.prune()

                if self.__running_model.empty:
                    LOGGER.debug("no events found in timeframe %s - %s",
                                 self.__running_model.start, self.__running_model.end)

            # Add the event to the running model
            LOGGER.spam("updating running model (timeframe %s - %s) with event %r",
                        self.__running_model.start, self.__running_model.end, event)
            self.__update_running_model(event)


@dataclass
class TimesPerCase:
    """
    The features that describe the reference and running models for a given change at a case level.

    Parameters
    ----------
    * `model`:    *the model containing the running and reference events for the drift*

    Returns
    -------
    * the per-case features
    """

    model: Drift
    """The drift model"""

    @cached_property
    def cycle_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the cycle time for the running and reference models, in seconds"""
        return Pair(
            reference=self.model.reference_durations,
            running=self.model.running_durations,
            unit="seconds",
        )

    @cached_property
    def processing_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the set of processing times for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.processing_time.total.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.processing_time.total.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def effective_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the effective processing time for each case for the running and reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.processing_time.effective.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.processing_time.effective.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def idle_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the idle processing time (the processing time when the resource is not available) for each case for the running and reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.processing_time.idle.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.processing_time.idle.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def waiting_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the set of waiting times for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.total.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.total.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def batching_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the part of waiting times due to batching for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.batching.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.batching.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def contention_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the part of waiting times due to contention for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.contention.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.contention.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def prioritization_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the part of waiting times due to prioritization for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.prioritization.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.prioritization.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def availability_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the part of waiting times due to resources unavailability for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.availability.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.availability.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )

    @cached_property
    def extraneous_time(self: typing.Self) -> Pair[typing.Iterable[int]]:
        """Get the part of waiting times due to extraneous factors for each activity for the running and the reference models"""
        return Pair(
            reference=sorted([
                sum(values) for values in _aggregate(
                    self.model.reference_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.extraneous.duration.total_seconds()),
                ).values()
            ]),
            running=sorted([
                sum(values) for values in _aggregate(
                    self.model.running_model,
                    key_extractor=lambda event: event.case,
                    value_extractor=lambda event: round(event.waiting_time.extraneous.duration.total_seconds()),
                ).values()
            ]),
            unit="seconds",
        )


@dataclass
class TimesPerActivity:
    """
    The features that describe the reference and running models for a given change at an activity level.

    Parameters
    ----------
    * `model`:    *the model containing the running and reference events for the drift*

    Returns
    -------
    * the per-activity features
    """

    model: Drift
    """The drift model"""

    @cached_property
    def processing_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """Get the set of processing times for each activity for the running and the reference models"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.processing_time.total.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.processing_time.total.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def effective_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.processing_time.effective.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.processing_time.effective.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def idle_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.processing_time.idle.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.processing_time.idle.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def waiting_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.total.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.total.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def batching_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.batching.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.batching.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def contention_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.contention.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.contention.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def prioritization_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.prioritization.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.prioritization.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def availability_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.availability.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.availability.duration.total_seconds()),
            ),
            unit="seconds",
        )

    @cached_property
    def extraneous_time(self: typing.Self) -> Pair[typing.Mapping[str, typing.Iterable[int]]]:
        """TODO docs"""
        return Pair(
            reference=_aggregate(
                self.model.reference_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.extraneous.duration.total_seconds()),
            ),
            running=_aggregate(
                self.model.running_model,
                key_extractor=lambda event: event.activity,
                value_extractor=lambda event: round(event.waiting_time.extraneous.duration.total_seconds()),
            ),
            unit="seconds",
        )
