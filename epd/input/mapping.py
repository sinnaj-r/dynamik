from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from epd.model import Event


@dataclass
class Mapping:
    """Define the mapping between a dict and Event attributes."""

    start: str
    end: str
    case: str
    activity: str
    resource: str

    __slots__ = ["start", "end", "case", "activity", "resource"]

    def __init__(self: Mapping,
                 start: str,
                 end: str,
                 case: str,
                 activity: str,
                 resource: str) -> None:
        """
        Create a dict <-> Event mapping.

        :param start: the name of the dict field containing the start timestamp for the event
        :param end: the name of the dict field containing the end timestamp for the event
        :param case: the name of the dict field containing the case for the event
        :param activity: the name of the dict field containing the activity for the event
        :param resource: the name of the dict field containing the resource for the event
        """
        self.start = start
        self.end = end
        self.case = case
        self.activity = activity
        self.resource = resource

    def dict_to_event(self: Mapping, source: dict[str, Any]) -> Event:
        """
        Create an Event instance from a source dictionary.

        :param source: a dictionary with the attributes of the event
        :return: an Event instance with the attributes initialized to the corresponding values in the source dictionary
        """
        instance = Event(
            case=source[self.case],
            activity=source[self.activity],
            start=source[self.start],
            end=source[self.end],
            resource=source[self.resource],
        )

        logging.debug('transforming %(source)r to %(instance)r', {'source': source, 'instance': instance})

        return instance
