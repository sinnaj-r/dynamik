from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta, datetime
from functools import cache as memoized
from itertools import dropwhile
from typing import Any

from fields import EventFields


@dataclass
class Event:
    case: str
    activity: str
    start: datetime
    end: datetime
    resource: str

    @classmethod
    def from_dict(cls, source: dict[str, Any], fields: EventFields) -> Event:
        """
            Creates an Event instance from a source dictionary

            :param source: a dictionary with the attributes of the event
            :param fields: an object with the mappings between the source dictionary keys and the event attributes
        """
        instance = cls(
            case=source[fields.CASE],
            activity=source[fields.ACTIVITY],
            start=source[fields.START],
            end=source[fields.END],
            resource=source[fields.RESOURCE],
        )

        logging.debug('transforming %(source)r to %(instance)r', {'source': source, 'instance': instance})

        return instance


class Model:
    """
        The class storing the model that will be used to detect drifts in the process.

        :param timeframe_size: the size of time period that will be analyzed on each iteration. Only needed if
                                      the Mode.AVERAGE is used.
        :param initial_activity: the activity that marks the start of a case.
        :param final_activity: the activity thar marks the final of a case.
    """

    def __init__(
            self,
            timeframe_size: timedelta | None = None,
            initial_activity: str = 'START',
            final_activity: str = 'END'
    ):
        self.__reference_model_start: datetime | None = None
        self.__reference_model_end: datetime | None = None
        self.__reference_model: list[Event] = []
        self.__reference_period_size = timeframe_size

        self.__running_model: list[Event] = []

        self.__initial_activity = initial_activity
        self.__final_activity = final_activity

        self.__cases: dict = {}

    @memoized
    def __compute_case_duration(self, case: str) -> timedelta:
        case_events: list[Event] = self.__cases[case]
        return max([evt.end for evt in case_events]) - min([evt.start for evt in case_events])

    def __is_case_complete(self, case: str) -> bool:
        case_events: list[Event] = self.__cases[case]
        return case_events[-1].activity == self.__final_activity

    def __update_cases(self, event: Event):
        # Add the event to the collection of running cases if the case exists...
        if (event.case in self.__cases) and (self.__cases[event.case][-1].activity != self.__final_activity):
            logging.debug('event %(event)r added to case %(case)s', {'event': event, 'case': event.case})
            self.__cases[event.case].append(event)
        # ...or create a new case if it does not exist and the event corresponds to the initial activity
        elif (event.case not in self.__cases) and (event.activity == self.__initial_activity):
            logging.debug('case %(case)s created', {'case': event.case})
            self.__cases[event.case] = [event]
        else:
            logging.debug('event %(event)r ignored', {'event': event})

    def __update_reference_model(self, event: Event):
        if self.__reference_model_start is None:
            self.__reference_model_start = event.start
            self.__reference_model_end = event.start + self.__reference_period_size

        if self.__reference_model_start <= event.start and event.end <= self.__reference_model_end:
            self.__reference_model.append(event)

    def __update_running_model(self, event: Event):
        self.__running_model.append(event)
        threshold = event.end - self.__reference_period_size
        self.__running_model = list(dropwhile(lambda evt: evt.start < threshold, self.__running_model))

    def update(self, event: Event) -> int | None:
        """
            Updates the model with a new event.
            If the case already exist and the final activity has not been reached, adds the event to the __cases
            dictionary.
            Else, if the event corresponds to the initial activity, a new case is created.
            Otherwise, the event is discarded

            :param event: the new event to be added to the model
            :return: the duration of the case corresponding the event, if the case is completed, or None otherwise.
        """
        self.__update_cases(event)
        self.__update_reference_model(event)
        self.__update_running_model(event)

        if (event.case in self.__cases.keys()) and self.__is_case_complete(event.case):
            # Return the case duration if completed and remove it from the cases' collection.
            case_duration = self.__compute_case_duration(event.case)
            del self.__cases[event.case]

            logging.debug('case %(case)s completed. case duration = %(duration)s',
                          {'case': event.case, 'duration': case_duration})
            return case_duration.total_seconds()

        else:
            return None

    def reset(self):
        """
            Resets the model forgetting all accumulated knowledge
        """
        self.__cases = {}
        self.__reference_model = []
        self.__reference_model_start = None
        self.__reference_model_end = None
        self.__running_model = []
