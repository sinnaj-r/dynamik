from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import dropwhile

from epd.model import Event


class Model:
    """
    The class storing the model that will be used to detect drifts in the process.

    :param timeframe_size: the size of time period that will be analyzed on each iteration.
    :param initial_activity: the activity that marks the start of a case.
    :param final_activity: the activity thar marks the final of a case.
    """

    __slots__ = ["__running_model", "__reference_model", "__reference_model_start", "__reference_model_end",
                 "__timeframe_size", "__initial_activity", "__final_activity", "__cases"]

    def __init__(
            self: Model,
            timeframe_size: timedelta | None = None,
            initial_activity: str = 'START',
            final_activity: str = 'END',
    ) -> None:
        """
        Create a new empty drift detection model with the given timeframe size and limit activities.

        :param timeframe_size: the timeframe used to build the reference and running models
        :param initial_activity: the activity marking the beginning of the subprocess to monitor
        :param final_activity: the activity marking the end of the subprocess to monitor
        """
        self.__reference_model_start: datetime | None = None
        self.__reference_model_end: datetime | None = None
        self.__reference_model: list[Event] = []
        self.__timeframe_size = timeframe_size

        self.__running_model: list[Event] = []

        self.__initial_activity = initial_activity
        self.__final_activity = final_activity

        self.__cases: dict = {}

    def __compute_case_duration(self: Model, case: str) -> timedelta:
        """
        Compute the duration of a case given his case identifier.

        The case duration is computed as ``max(end time) - min(start time)``

        :param case: the case identifier
        :return: the case duration
        """
        return max(evt.end for evt in self.__cases[case]) - min(evt.start for evt in self.__cases[case])

    def __is_case_complete(self: Model, case: str) -> bool:
        """
        Check if a case completed its execution.

        :param case: the case identifier
        :return: a boolean indicating if the case execution has been completed or not
        """
        case_events: list[Event] = self.__cases[case]
        return case_events[-1].activity == self.__final_activity

    def __update_cases(self: Model, event: Event) -> None:
        """
        Update cases with a new event, appending it to his respective case from self.__cases.

        The event is added only if it belongs to a valid execution:
            - The case must not be complete (i.e., the last event for the case does not match ``self.__final_activity``)
            - If the case does not exist, it will be created only if the event activity matches
              ``self.__initial_activity``
        If none of the conditions is satisfied, the event gets discarded

        :param event: the new event that will be added to the cases
        """
        # Add the event to the collection of running cases if the case exists...
        if (event.case in self.__cases) and (self.__cases[event.case][-1].activity != self.__final_activity):
            logging.debug('event %(event)r added to case %(case)s',
                          {'event': event, 'case': event.case})
            self.__cases[event.case].append(event)
        # or create a new case if it doesn't exist and the event corresponds to the initial activity
        elif (event.case not in self.__cases) and (event.activity == self.__initial_activity):
            logging.debug('case %(case)s created', {'case': event.case})
            self.__cases[event.case] = [event]
        else:
            logging.debug('event %(event)r ignored', {'event': event})

    def __update_reference_model(self: Model, event: Event) -> None:
        """
        Update the reference model with a new event.

        The reference model will be updated if its empty or if the event completed its execution before
        ``self.__reference_model_end``.
        If the reference model is empty, ``self.__reference_model_start`` and ``self.__reference_model_end`` are also
        initialized:
            - ``self.__reference_model_start = event.start``
            - ``self.__reference_model_end = event.start + self.__timeframe_size``

        :param event: the new event that will be used to update the reference model
        """
        if self.__reference_model_start is None:
            self.__reference_model_start = event.start
            self.__reference_model_end = event.start + self.__timeframe_size

        if self.__reference_model_start <= event.start and event.end <= self.__reference_model_end:
            self.__reference_model.append(event)

    def __update_running_model(self: Model, event: Event) -> None:
        """
        Update the running model with a new event.

        When the running model is updated with a new event, events that started before than
        ``event.end - self.__timeframe_size`` are dropped, so only the last ``self.__timeframe_size`` is used as the
        current model.

        :param event: the new event used to update the running model
        """
        self.__running_model.append(event)
        self.__running_model = list(
            dropwhile(lambda evt: evt.start < (event.end - self.__timeframe_size), self.__running_model),
        )

    @property
    def reference_model(self: Model) -> Iterable[Event]:
        """Get the list of events that are used as a reference model against which changes are checked."""
        return self.__reference_model

    @property
    def running_model(self: Model) -> Iterable[Event]:
        """Get the list of events used as the running model being checked for changes."""
        return self.__running_model

    def update(self: Model, event: Event) -> float | None:
        """
        Update the model with a new event.

        Updates the model with a new event. If the case already exist and the final activity has not
        been reached, adds the event to the self.__cases dictionary.
        Else, if the event corresponds to the initial activity, a new case is created.
        Otherwise, the event is discarded.

        :param event: the new event to be added to the model
        :return: the duration of the case corresponding the event, if the case is completed, or
        None otherwise.
        """
        self.__update_cases(event)
        self.__update_reference_model(event)
        self.__update_running_model(event)

        if (event.case in self.__cases) and self.__is_case_complete(event.case):
            # Return the case duration if completed and remove it from the cases' collection.
            case_duration = self.__compute_case_duration(event.case)
            del self.__cases[event.case]

            logging.debug('case %(case)s completed. case duration = %(duration)s',
                          {'case': event.case, 'duration': case_duration})
            return case_duration.total_seconds()

        return None

    def reset(self: Model) -> None:
        """Reset the model forgetting all accumulated knowledge."""
        self.__cases = {}
        self.__reference_model = []
        self.__reference_model_start = None
        self.__reference_model_end = None
        self.__running_model = []
