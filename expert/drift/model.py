from __future__ import annotations

import enum
import typing
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean

from expert.__logger import LOGGER
from expert.model import Event

T = typing.TypeVar("T")

@dataclass
class Drift:
    """TODO"""

    level: DriftLevel
    reference_model: typing.Iterable[Event] | None = None
    running_model: typing.Iterable[Event] | None = None
    reference_durations: typing.Iterable[float] | None = None
    running_durations: typing.Iterable[float] | None = None


class DriftLevel(enum.Enum):
    """The drift level. Can be no drift, drift warning or confirmed drift."""

    NONE = 0
    WARNING = 1
    CONFIRMED = 2


NO_DRIFT=Drift(level=DriftLevel.NONE)


@dataclass
class _Pair(typing.Generic[T]):
    reference: T
    running: T


@dataclass
class DriftCauses:
    """
    The causes of a drift.

    It contains the corresponding running and reference collections of events, the computed arrival rates for each
    activity, the resource utilization rates, the mean waiting times...
    """

    model: _Pair[typing.Iterable[Event]]
    """The pair of collections of events used as the reference and running models"""
    case_duration: _Pair[typing.Iterable[float]]
    """The case durations for the running and reference models"""
    arrival_rate: _Pair[typing.Mapping[str, typing.Iterable[float]]]
    """The arrival rate for each activity for the running and the reference models"""
    resource_utilization_rate: _Pair[typing.Mapping[str, typing.Iterable[float]]]
    """The utilization rate for each resource for the running and the reference models"""
    waiting_time: _Pair[typing.Mapping[str, typing.Iterable[timedelta]]]
    """The mean waiting time for each activity for the running and the reference models"""
    arrival_rate_changed: bool
    """A flag showing if there are significant differences between the reference and the running arrival rates"""
    resource_utilization_rate_changed: bool
    """A flag showing if there are significant differences between the reference and the running utilization rates """
    waiting_time_changed: bool
    """A flag showing if there are significant differences between the reference and the running waiting times """


class DriftModel:
    """Stores the model that will be used to detect drifts in the process."""

    # The size of the reference and running models, in time units
    __timeframe_size: timedelta
    # The initial activities of the process
    __initial_activities: typing.Iterable[str]
    # The final activities of the process
    __final_activities: typing.Iterable[str]
    # The number of drift warnings to confirm a drift
    __warnings_to_confirm: int = 0
    # The period considered as a warm-up
    __warm_up: timedelta
    # The statistical test used to determine if the model presents a drift
    __test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool]
    # The overlap between running models
    __overlap: timedelta = timedelta()
    # The collection of events used as the reference model
    __reference_model: typing.MutableSequence[Event] = []
    # The collection of events used as the running model
    __running_model: typing.MutableSequence[Event] = []
    # The date and time when the reference model starts
    __reference_model_start: datetime | None = None
    # The date and time when the reference model ends
    __reference_model_end: datetime | None = None
    # The date and time when the running model starts
    __running_model_start: datetime | None = None
    # The date and time when the running model ends
    __running_model_end: datetime | None = None
    # The start date and time for each complete case in the reference model
    __reference_cases_start: typing.MutableMapping[str, datetime] = {}
    # The start date and time for each complete case in the running model
    __running_cases_start: typing.MutableMapping[str, datetime] = {}
    # The duration of each complete case in the reference model
    __reference_durations: typing.MutableMapping[str, float] = {}
    # The duration of each complete case in the running model
    __running_durations: typing.MutableMapping[str, float] = {}
    # The collection of detection results
    __drifts: typing.MutableSequence[Drift] = deque([NO_DRIFT], maxlen=1)


    def __init__(
            self: typing.Self,
            *,
            timeframe_size: timedelta,
            test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool],
            initial_activities: typing.Iterable[str] = tuple("START"),
            final_activities: typing.Iterable[str] = tuple("END"),
            warm_up: timedelta = timedelta(),
            overlap_between_models: timedelta = timedelta(),
            warnings_to_confirm: int = 3,
    ) -> None:
        """
        Create a new empty drift detection model with the given timeframe size and limit activities.

        Parameters
        ----------
        * `timeframe_size`:         *the timeframe used to build the reference and running models*
        * `initial_activities`:     *the list of activities marking the beginning of the subprocess to monitor*
        * `final_activities`:       *the list of activities marking the end of the subprocess to monitor*
        * `warm_up`:                *the warm-up period during which events will be discarded*
        * `overlap_between_models`: *the overlapping between running models (must be smaller than the timeframe size)*
        * `test`:                   *the test used for evaluating if there are any difference between the reference and
                                     the running models*
        * `warnings_to_confirm`:    *the number of consecutive detections needed for confirming a drift*
        """
        self.__timeframe_size = timeframe_size
        self.__initial_activities = initial_activities
        self.__final_activities = final_activities
        self.__warm_up = warm_up
        self.__test = test
        self.__warnings_to_confirm = warnings_to_confirm
        self.__drifts = deque([NO_DRIFT] * warnings_to_confirm, maxlen=warnings_to_confirm)
        self.__overlap = overlap_between_models


    @property
    def reference_model(self: typing.Self) -> typing.Iterable[Event]:
        """The list of events that are used as a reference model against which changes are checked"""
        return self.__reference_model


    @property
    def running_model(self: typing.Self) -> typing.Iterable[Event]:
        """The list of events used as the running model being checked for changes"""
        return self.__running_model


    @property
    def reference_model_durations(self: typing.Self) -> typing.Mapping[str, float] | None:
        """The mapping of (case id, duration) for the reference model"""
        return self.__reference_durations


    @property
    def running_model_durations(self: typing.Self) -> typing.Mapping[str, float] | None:
        """The mapping of (case id, duration) for the running model"""
        return self.__running_durations


    @property
    def drift(self: typing.Self) -> Drift:
        """The last iteration drift status"""
        return self.__drifts[-1]


    def __update_reference_model(self: typing.Self, event: Event) -> None:
        LOGGER.debug("updating reference model")
        # Append the event to the list of events in the reference model
        self.__reference_model.append(event)
        # Store the case start timestamp
        if event.activity in self.__initial_activities:
            LOGGER.spam("adding case %s to reference model (timeframe %s - %s)",
                          event.case, self.__reference_model_start, self.__reference_model_end)
            self.__reference_cases_start[event.case] = event.start
        # If the event is the final activity and the case started in the reference timeframe, compute the case duration
        if event.activity in self.__final_activities and event.case in self.__reference_cases_start:
            LOGGER.spam("adding case %s duration to reference model (timeframe %s - %s)",
                          event.case, self.__reference_model_start, self.__reference_model_end)
            case_start = self.__reference_cases_start[event.case]
            self.__reference_durations[event.case] = (event.end - case_start).total_seconds()


    def __update_running_model(self: typing.Self, event: Event) -> None:
        LOGGER.debug("updating running model")
        # Update events in the running model
        self.__running_model.append(event)
        # Store the case start if the event corresponds to the initial activity
        if event.activity in self.__initial_activities:
            LOGGER.spam("adding case %s to running model (timeframe %s - %s)",
                          event.case, self.__reference_model_start, self.__reference_model_end)
            self.__running_cases_start[event.case] = event.start
        # If the event is the final activity and the case started in the running timeframe, compute the case duration
        if event.activity in self.__final_activities and event.case in self.__running_cases_start:
            LOGGER.spam("adding case %s duration to running model (timeframe %s - %s)",
                          event.case, self.__reference_model_start, self.__reference_model_end)
            case_start = self.__running_cases_start[event.case]
            self.__running_durations[event.case] = (event.end - case_start).total_seconds()


    def __prune_running(self: typing.Self) -> None:
        LOGGER.debug("pruning running log")
        # Remove all events that are out of the overlapping region from the running model
        while len(self.__running_model) > 0 and self.__running_model[0].start < self.__running_model_start:
            # Remove the event from the model
            outdated = self.__running_model.pop(0)
            LOGGER.spam("removing event %r from running model (timeframe %s - %s)",
                          outdated, self.__reference_model_start, self.__reference_model_end)
            # Remove the case start for the outdated event
            if outdated.case in self.__running_cases_start:
                LOGGER.spam("removing case %s from running model (timeframe %s - %s)",
                              outdated.case, self.__reference_model_start, self.__reference_model_end)
                del self.__running_cases_start[outdated.case]
            # Remove the case duration for the outdated event
            if outdated.case in self.__running_durations:
                LOGGER.spam("removing case %s duration from running model (timeframe %s - %s)",
                              outdated.case, self.__reference_model_start, self.__reference_model_end)
                del self.__running_durations[outdated.case]


    def __update_drifts(self: typing.Self) -> None:
        LOGGER.debug("updating drifts")
        if len(self.__running_durations.values()) == 0:
            LOGGER.warning("no case executed completely in the running timeframe %s - %s",
                            self.__running_model_start,
                            self.__running_model_end)
            return

        # If the model presents a drift add a warning
        if self.__test(list(self.__reference_durations.values()), list(self.__running_durations.values())):
            self.__drifts.append(Drift(
                level=DriftLevel.WARNING,
                reference_model=tuple(self.reference_model),
                running_model=tuple(self.running_model),
                reference_durations=tuple(sorted(self.__reference_durations.values())),
                running_durations=tuple(sorted(self.__running_durations.values())),
            ))
            LOGGER.verbose(
                "drift warning between reference timeframe (%s - %s) -> %s and running timeframe (%s, %s) -> %s",
                self.__reference_model_start, self.__reference_model_end,
                mean(list(self.__reference_durations.values())),
                self.__running_model_start, self.__running_model_end,
                mean(list(self.__running_durations.values())))
            # If all the detections are warnings, replace by a confirmation
            if all(drift.level == DriftLevel.WARNING for drift in self.__drifts):
                LOGGER.verbose(
                    "drift confirmed between reference timeframe (%s - %s) -> %s and running timeframe (%s, %s) -> %s",
                    self.__reference_model_start, self.__reference_model_end,
                    mean(list(self.__reference_durations.values())),
                    self.__running_model_start, self.__running_model_end,
                    mean(list(self.__running_durations.values())),
                )
                self.__drifts.pop()
                self.__drifts.append(Drift(
                    level=DriftLevel.CONFIRMED,
                    reference_model=tuple(self.reference_model),
                    running_model=tuple(self.running_model),
                    reference_durations=tuple(sorted(self.__reference_durations.values())),
                    running_durations=tuple(sorted(self.__running_durations.values())),
                ))
        # If the model does not present a drift add a NONE
        else:
            LOGGER.verbose("no drift between reference timeframe (%s - %s) -> %s and running timeframe (%s, %s) -> %s",
                         self.__reference_model_start, self.__reference_model_end,
                         mean(list(self.__reference_durations.values())),
                         self.__running_model_start, self.__running_model_end,
                         mean(list(self.__running_durations.values())))
            self.__drifts.append(NO_DRIFT)


    def reset(self: typing.Self) -> None:
        """Reset the model to the initial state"""
        self.__reference_model = []
        self.__running_model = []

        self.__reference_model_start = None
        self.__running_model_start = None

        self.__reference_model_end = None
        self.__running_model_end = None

        self.__reference_cases_start = {}
        self.__running_cases_start = {}

        self.__reference_durations = {}
        self.__running_durations = {}

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
        if self.__reference_model_start is None:
            # Set the reference model start and end dates
            self.__reference_model_start = event.start + self.__warm_up
            self.__reference_model_end = self.__reference_model_start + self.__timeframe_size
            LOGGER.debug("updating reference model to timeframe (%s - %s)",
                          self.__reference_model_start, self.__reference_model_end)
        # Update the running model timeframe if it is not initialized
        if self.__running_model_start is None:
            # Set the running model start and end
            self.__running_model_start = self.__reference_model_end - self.__overlap
            self.__running_model_end = self.__running_model_start + self.__timeframe_size
            LOGGER.debug("updating running model to timeframe (%s - %s)",
                          self.__running_model_start, self.__running_model_end)
        # Drop the event if it is part of the warm-up period
        if event.start < self.__reference_model_start:
            LOGGER.spam("dropping warm-up event %r", event)
        # If the event is part of the reference model, update it
        if (self.__reference_model_start <= event.start) and (event.end <= self.__reference_model_end):
            LOGGER.debug("updating reference model (timeframe %s - %s) with event %r",
                          self.__reference_model_start, self.__reference_model_end, event)
            self.__update_reference_model(event)
        # If the event is part of the running model, update it
        if (self.__running_model_start <= event.start) and (event.end <= self.__running_model_end):
            LOGGER.spam("updating running model (timeframe %s - %s) with event %r",
                          self.__reference_model_start, self.__reference_model_end, event)
            self.__update_running_model(event)
        # If the event is out of the running model, update the limits and prune the running model content
        if event.end > self.__running_model_end:
            if len(self.__running_model) == 0:
                LOGGER.warning("no event executed in the timeframe %s - %s",
                                self.__running_model_start,
                                self.__running_model_end)
            else:
                # Update the drifts
                LOGGER.verbose("checking drift between reference timeframe %s - %s and running timeframe %s - %s",
                             self.__reference_model_start, self.__reference_model_end,
                             self.__running_model_start, self.__running_model_end)
                self.__update_drifts()

            # Update the running model limits
            self.__running_model_start = self.__running_model_end - self.__overlap
            self.__running_model_end = self.__running_model_start + self.__timeframe_size
            LOGGER.debug("updating running model to timeframe (%s - %s)",
                          self.__running_model_start, self.__running_model_end)

            # Delete outdated events from the running model
            LOGGER.debug("pruning running model (timeframe %s - %s)",
                          self.__reference_model_start, self.__reference_model_end)
            self.__prune_running()
            # Add the event to the running model
            LOGGER.debug("updating running model (timeframe %s - %s) with event %r",
                          self.__reference_model_start, self.__reference_model_end, event)
            self.__update_running_model(event)
