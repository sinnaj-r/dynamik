from __future__ import annotations

import logging
import sys
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta

from expert.model import Event


class Model:
    """Stores the model that will be used to detect drifts in the process."""

    __timeframe_size: timedelta
    __initial_activity: str
    __final_activity: str

    __reference_model: typing.MutableSequence[Event] = []
    __running_model: typing.MutableSequence[Event] = []

    __reference_model_ready: bool = False

    __reference_model_start: datetime | None = None
    __reference_model_end: datetime | None = None
    __running_model_start: datetime | None = None
    __running_model_end: datetime | None = None

    __reference_cases_start: typing.MutableMapping[str, datetime] = {}
    __running_cases_start: typing.MutableMapping[str, datetime] = {}

    __reference_durations: typing.MutableMapping[str, float] = {}
    __running_durations: typing.MutableMapping[str, float] = {}

    def __init__(
            self: typing.Self,
            timeframe_size: timedelta | None = None,
            *,
            initial_activity: str = 'START',
            final_activity: str = 'END',
    ) -> None:
        """
        Create a new empty drift detection model with the given timeframe size and limit activities.

        Parameters
        ----------
        * `timeframe_size`:     *the timeframe used to build the reference and running models*
        * `initial_activity`:   *the activity marking the beginning of the subprocess to monitor*
        * `final_activity`:     *the activity marking the end of the subprocess to monitor*
        """
        self.__timeframe_size = timeframe_size
        self.__initial_activity = initial_activity
        self.__final_activity = final_activity

    @property
    def model_ready(self: typing.Self) -> bool:
        """A boolean indicating if the reference model is ready to be used"""
        return self.__reference_model_end < self.__running_model_end \
            and len(self.__reference_model) > 0 \
            and len(self.__running_model) > 0

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

    def __update_reference_model(self: typing.Self, event: Event) -> None:
        # Initialize the reference model time limits if they are not initialized
        if self.__reference_model_start is None:
            self.__reference_model_start = event.start
            self.__reference_model_end = event.start + self.__timeframe_size
        # Update events in the reference model
        if event.end < self.__reference_model_end:
            self.__reference_model.append(event)
            # Store the case start timestamp
            if event.activity == self.__initial_activity:
                self.__reference_cases_start[event.case] = event.start
            # If the event is the final activity and the case is still within the timeframe, compute the case duration
            if event.activity == self.__final_activity and event.case in self.__reference_cases_start:
                self.__reference_durations[event.case] = (event.end - self.__reference_cases_start[event.case]).total_seconds()

    def __update_running_model(self: typing.Self, event: Event) -> None:
        # Update running model time limits
        self.__running_model_start = event.end - self.__timeframe_size
        self.__running_model_end = event.end
        # Update events in the running model
        self.__running_model.append(event)
        # Store the case start if the event corresponds to the initial activity
        if event.activity == self.__initial_activity:
            self.__running_cases_start[event.case] = event.start
        # Drop events out of the timeframe
        while len(self.__running_model) > 0 and self.__running_model[0].start < self.__running_model_start:
            outdated_event = self.__running_model.pop(0)
            # If the outdated event was the first activity from a case, the case is outdated too
            if outdated_event.activity == self.__initial_activity:
                # Remove the outdated case from the runninf cases
                del self.__running_cases_start[outdated_event.case]
                # Remove the outdated case from the running durations if it is present
                # (it can be not present in the durations if the case becomes outdated after finishing its execution)
                if outdated_event.case in self.__running_durations:
                    del self.__running_durations[outdated_event.case]
        # Compute the case duration if the event is the last activity from a case and the case is running
        if event.activity == self.__final_activity and event.case in self.__running_cases_start:
            self.__running_durations[event.case] = (event.end - self.__running_cases_start[event.case]).total_seconds()

    def update(self: typing.Self, event: Event) -> None:
        """
        Update the model with a new event.

        If the event lies in the reference time window it will be added to both the reference and the running model.
        Otherwise, only the running model will be updated.

        Parameters
        ----------
        * `event`: *the new event to be added to the model*
        """
        self.__update_reference_model(event)
        self.__update_running_model(event)

        if len(self.__reference_cases_start) == 0:
            logging.warning("no cases execute completely within the given reference timeframe. try increasing it")
            sys.exit("NO_CASES_IN_REFERENCE_TIMEFRAME")
        if len(self.__running_cases_start) == 0:
            logging.warning("no cases execute completely within the given running timeframe. try increasing it")
            sys.exit("NO_CASES_IN_RUNNING_TIMEFRAME")

T = typing.TypeVar("T")

@dataclass
class Pair(typing.Generic[T]):
    """A class representing a pair of T objects, one for the reference and one for the running model."""

    reference: T
    """The value associated to the reference model"""
    running: T
    """The value associated to the running model"""


@dataclass
class Result:
    """
    The result of a concept drift detection.

    It contains the corresponding running and reference collections of events, the computed arrival rates for each
    activity, the resource utilization rates, the mean waiting times...
    """

    model: Pair[typing.Iterable[Event]]
    """The pair of collections of events used as the reference and running models"""
    case_duration: Pair[typing.Mapping[str, float]]
    """The per case duration for the running and reference models"""
    arrival_rate: Pair[typing.Mapping[str, float]]
    """The arrival rate for each activity for the running and the reference models"""
    resource_utilization_rate: Pair[typing.Mapping[str, float]]
    """The utilization rate for each resource for the running and the reference models"""
    waiting_time: Pair[typing.Mapping[str, float]]
    """The mean waiting time for each activity for the running and the reference models"""
    arrival_rate_changed: bool
    """A flag showing if there are differences between the significative reference and the running arrival rates"""
    resource_utilization_rate_changed: bool
    """A flag showing if there are differences between the significative reference and the running utilization rates """
    waiting_time_changed: bool
    """A flag showing if there are differences between the significative reference and the running waiting times """


