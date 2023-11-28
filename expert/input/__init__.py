"""
This module contains the definitions for reading the logs later used by the drift detection algorithm.

Log parsers are expected to return a generator, which will be consumed later by the detection algorithm.
"""
from __future__ import annotations

import typing
from collections import namedtuple
from dataclasses import dataclass, field

from expert.logger import LOGGER
from expert.model import Event


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
    attributes: typing.Mapping[str, typing.Callable[[str], typing.Any] | bool] | typing.Iterable[str] = field(default_factory=frozenset)
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
        additional_attrs = {}

        for attr in self.attributes:
            # we force the attribute to be lowercase to maintain consistency with the rest of attributes
            attr = attr.lower()

            if (
                    # if self.attributes is a list of strings,
                    isinstance(self.attributes, typing.Iterable) or
                    # or if it is a mapping and the value is True, just add the value
                    (isinstance(self.attributes[attr], bool) and self.attributes[attr])
            ):
                additional_attrs[attr] = getattr(source, attr)
            # otherwise self.attributes is a mapping and the value is a function, so apply the transform to the value and add it
            elif isinstance(self.attributes[attr], typing.Callable):
                additional_attrs[attr] = self.attributes[attr](getattr(source, attr))

        instance = Event(
            case=getattr(source, self.case.lower()),
            activity=getattr(source, self.activity.lower()),
            start=getattr(source, self.start.lower()).to_pydatetime(),
            end=getattr(source, self.end.lower()).to_pydatetime(),
            resource=getattr(source, self.resource.lower()) if self.resource is not None else None,
            enabled=getattr(source, self.enablement.lower()).to_pydatetime() if self.enablement is not None else None,
            attributes=additional_attrs,
        )

        LOGGER.spam("transforming %(source)r to %(instance)r", {"source": source, "instance": instance})

        return instance
