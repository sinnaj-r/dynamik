"""
This module contains the definitions for reading the logs later used by the drift detection algorithm.

Log parsers are expected to return a generator, which will be consumed later by the detection algorithm.
"""
from __future__ import annotations

import json
import typing
from collections import namedtuple
from dataclasses import dataclass, field

from expert.model import Event
from expert.utils.logger import LOGGER


@dataclass
class EventMapping:
    """Defines the mapping between a dictionary-like object and Event attributes."""

    start: str
    """The attribute from the log file containing the event start timestamp"""
    end: str
    """The attribute from the log file containing the event finish timestamp"""
    case: str
    """The attribute from the log file containing the case ID"""
    activity: str
    """The attribute from the log file containing the activity name"""
    resource: str | None = None
    """The attribute from the log file containing the resource name"""
    enablement: str | None = None
    """The attribute from the log file containing the event enablement timestamp"""
    attributes: typing.Mapping[str, str] = field(default_factory=dict)
    """The set of additional attributes to be included when parsing the log"""

    def tuple_to_event(self: typing.Self, source: namedtuple) -> Event:
        """
        Create an `expert.model.Event` instance from a source *dictionary-like* object applying the current mapping.

        Parameters
        ----------
        * `source`: *a pandas Series with all the attributes of the event*

        Returns
        -------
        * the `expert.model.Event` instance resulting from applying this mapping to the *dictionary-like* source
        """
        instance = Event(
            case=getattr(source, self.case.lower()),
            activity=getattr(source, self.activity.lower()),
            start=getattr(source, self.start.lower()),
            end=getattr(source, self.end.lower()),
            resource=getattr(source, self.resource.lower()) if self.resource is not None else None,
            enabled=getattr(source, self.enablement.lower()) if self.enablement is not None else None,
            attributes={
                attr.lower(): getattr(source, attr_in_df.lower()) for (attr, attr_in_df) in self.attributes.items()
            },
        )

        LOGGER.spam("transforming %(source)r to %(instance)r", {"source": source, "instance": instance})

        return instance

    @staticmethod
    def parse(filepath: str) -> EventMapping:
        """Parse an event mapping from a JSON file"""
        with open(filepath) as _file:
            source = json.load(_file)

            return EventMapping(
                start=source["start"],
                end=source["end"],
                enablement=source["enablement"] if "enablement" in source else None,
                resource=source["resource"],
                activity=source["activity"],
                case=source["case"],
                attributes=source["attributes"] if "attributes" in source else {},
            )
