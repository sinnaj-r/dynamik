from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    """An object representing an event in a log."""

    case: str
    activity: str
    start: datetime
    end: datetime
    resource: str

    def __init__(self: Event,
                 *,
                 case: str,
                 activity: str,
                 start: datetime,
                 end: datetime,
                 resource: str) -> None:
        """
        Create a new Event given its attributes.

        :param case: the case identifier
        :param activity: the activity identifier
        :param start: the timestamp when the event started
        :param end: the timestamp when the event finished
        :param resource: the resource executing the activity
        """
        self.case = case
        self.activity = activity
        self.start = start
        self.end = end
        self.resource = resource
