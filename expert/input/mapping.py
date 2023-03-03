from __future__ import annotations

import collections.abc
import logging
import typing
from dataclasses import dataclass

from expert.model import Event


@dataclass
class Mapping:
    """Defines the mapping between a dictionary-like object and Event attributes."""

    start: str
    """The attribute from the log file containing the event start timestamp"""
    end: str
    """The attribute from the log file containing the event finish timestamp"""
    case: str
    """The attribute from the log file containing the case ID"""
    activity: str
    """The attribute from the log file containing the activity name"""
    resource: str
    """The attribute from the log file containing the resource name"""

    __slots__ = ["start", "end", "case", "activity", "resource"]

    def __init__(self: Mapping,
                 start: str,
                 end: str,
                 case: str,
                 activity: str,
                 resource: str) -> None:
        """
        Create a new mapping for transforming dictionary-like objects to `Event` objects.

        Parameters
        ----------
        * `start`:      *the name of the dict field containing the start timestamp for the events*
        * `end`:        *the name of the dict field containing the end timestamp for the events*
        * `case`:       *the name of the dict field containing the case for the events*
        * `activity`:   *the name of the dict field containing the activity for the events*
        * `resource`:   *the name of the dict field containing the resource for the events*
        """
        self.start = start
        self.end = end
        self.case = case
        self.activity = activity
        self.resource = resource

    def dict_to_event(self: Mapping, source: dict[str, typing.Any] | collections.abc.Mapping[str, typing.Any]) -> Event:
        """
        Create an `expert.model.Event` instance from a source *dictionary-like* object applying the current mapping.

        Parameters
        ----------
        * `source`: *a dictionary-like object with the attributes of the event*

        Returns
        -------
        * the `expert.model.Event` instance resulting from applying this mapping to the *dictionary-like* source
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
