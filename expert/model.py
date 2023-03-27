"""This module contains the definition of a model for the events from an event log."""
from __future__ import annotations

import typing
from dataclasses import dataclass
from datetime import datetime, timedelta


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
    enabled: datetime | None = None
    """The time when the activity was made available for execution"""

    @property
    def execution_time(self: typing.Self) -> timedelta:
        """The execution time elapsed between the event start end the event finalization"""
        return self.end - self.start

    @property
    def waiting_time(self: typing.Self) -> timedelta:
        """The waiting time between the activity enablement and the beginning of its execution"""
        return self.start - self.enabled

    @property
    def total_time(self: typing.Self) -> timedelta:
        """The total event time, between its enablement and the finalization of the execution"""
        return self.waiting_time + self.execution_time

    @property
    def violations(self: typing.Self) -> typing.Iterable[str]:
        """Get the violations of the validity of the event"""
        result = []
        if self.enabled > self.start:
            result.append("enabled > start")
        if self.enabled > self.end:
            result.append("enabled > end")
        if self.start > self.end:
            result.append("start > end")


        return result

    def is_valid(self: typing.Self) -> bool:
        """Check if the event is valid or malformed"""
        return self.enabled <= self.start <= self.end
