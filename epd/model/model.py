from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    """
    `Event` provides a standardized representation of an event from a log.

    An event in from a process log represents the execution of an activity in a specific instant in time by a specific
    resource in the context of a specific process execution.
    """

    case: str
    """The case identifier for the event, which associates it with a specific process execution"""
    activity: str
    """The activity being executed"""
    start: datetime
    """The time when the activity execution began"""
    end: datetime
    """The time when the activity execution ended"""
    resource: str
    """The resource in charge of the activity"""

    def __init__(self: Event, *, case: str, activity: str, start: datetime, end: datetime, resource: str) -> None:
        """
        Create a new Event instance given its attributes.

        Parameters
        ----------
        * `case`:     *the case identifier*
        * `activity`: *the activity identifier*
        * `start`:    *the timestamp when the event started*
        * `end`:      *the timestamp when the event finished*
        * `resource`: *the resource executing the activity*
        """
        self.case = case
        self.activity = activity
        self.start = start
        self.end = end
        self.resource = resource
