from __future__ import annotations

import logging
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import dropwhile

from expert.model import Event
from expert.utils import compute_case_duration, compute_enablement_timestamps


class Model:
    """Stores the model that will be used to detect drifts in the process."""

    __timeframe_size: timedelta
    __initial_activity: str
    __final_activity: str
    __cases: dict = {}
    __reference_model_start: datetime | None = None
    __reference_model_end: datetime | None = None
    __reference_model: list[Event] = []
    __running_model: list[Event] = []

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
    def reference_model(self: typing.Self) -> typing.Iterable[Event]:
        """The list of events that are used as a reference model against which changes are checked"""
        return compute_enablement_timestamps(list(self.__reference_model))

    @property
    def running_model(self: typing.Self) -> typing.Iterable[Event]:
        """The list of events used as the running model being checked for changes"""
        return compute_enablement_timestamps(list(self.__running_model))

    def __is_case_complete(self: typing.Self, case: str) -> bool:
        # Check if the case is complete by looking at its last activity
        return self.__cases[case][-1].activity == self.__final_activity

    def __update_cases(self: typing.Self, event: Event) -> None:
        # Add the event to the collection of running cases if the case exists...
        if (event.case in self.__cases) and (self.__cases[event.case][-1].activity != self.__final_activity):
            logging.debug('event %(event)r added to case %(case)s',
                          {'event': event, 'case': event.case})
            self.__cases[event.case].append(event)
        # ...or create a new case if it doesn't exist and the event corresponds to the initial activity
        elif (event.case not in self.__cases) and (event.activity == self.__initial_activity):
            logging.debug('case %(case)s created', {'case': event.case})
            self.__cases[event.case] = [event]
        # Else skip this event
        else:
            logging.debug('event %(event)r ignored', {'event': event})

    def __update_reference_model(self: typing.Self, event: Event) -> None:
        # Initialize the reference model time limits if they are not initialized
        if self.__reference_model_start is None:
            self.__reference_model_start = event.start
            self.__reference_model_end = event.start + self.__timeframe_size
        # Add the event to the reference model if it is between the start and end limits
        if self.__reference_model_start <= event.start and event.end <= self.__reference_model_end:
            self.__reference_model.append(event)

    def __update_running_model(self: typing.Self, event: Event) -> None:
        # Add the new event to the running model
        self.__running_model.append(event)
        # Remove the events that are out of the reference model time limits
        self.__running_model = list(
            dropwhile(lambda evt: evt.start < (event.end - self.__timeframe_size), self.__running_model),
        )

    def update(self: typing.Self, event: Event) -> float | None:
        """
        Update the model with a new event.

        If the case already exist and the final activity has not been reached, adds the event to the `cases` dictionary.
        Else, if the event corresponds to the initial activity, a new case is created. Otherwise, event is discarded.

        Parameters
        ----------
        * `event`: *the new event to be added to the model*

        Returns
        -------
        * the duration of the case corresponding the event, if the case is completed, and `None` otherwise
        """
        # Update the models with the new event
        self.__update_cases(event)
        self.__update_reference_model(event)
        self.__update_running_model(event)

        if (event.case in self.__cases) and self.__is_case_complete(event.case):
            # Return the case duration if completed and remove it from the cases' collection.
            case_duration = compute_case_duration(self.__cases[event.case])
            del self.__cases[event.case]

            logging.debug('case %(case)s completed. case duration = %(duration)s',
                          {'case': event.case, 'duration': case_duration})
            return case_duration.total_seconds()

        return None

    def reset(self: typing.Self) -> None:
        """Reset the model forgetting all accumulated knowledge."""
        self.__cases = {}
        self.__reference_model = []
        self.__reference_model_start = None
        self.__reference_model_end = None
        self.__running_model = []


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
    arrival_rate: Pair[typing.Mapping[str, float]]
    """The arrival rate for each activity for the running and the reference models"""
    resource_utilization_rate: Pair[typing.Mapping[str, float]]
    """The utilization rate for each resource for the running and the reference models"""
    waiting_time: Pair[typing.Mapping[str, float]]
    """The mean waiting time for each activity for the running and the reference models"""


