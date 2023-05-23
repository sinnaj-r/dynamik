"""This module contains the definition of a model for the events from an event log."""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property

from intervaltree import Interval


@dataclass
class Batch:
    """A batch descriptor"""

    activity: str
    """The activity performed during the batch"""
    resource: str
    """The resource executing the batch"""
    size: int
    """The batch size"""
    accumulation: Interval[datetime]
    """The batch accumulation interval"""
    execution: Interval[datetime]
    """The batch execution interval"""


@dataclass
class IntervalTime:
    """TODO"""

    intervals: typing.Iterable[Interval] = ()
    duration: timedelta = timedelta()


@dataclass
class WaitingTime:
    """An object representing the waiting time for an event, with its decomposition"""

    event: Event

    batching: IntervalTime = field(default_factory= IntervalTime, init=False)
    contention: IntervalTime = field(default_factory= IntervalTime, init=False)
    prioritization: IntervalTime = field(default_factory= IntervalTime, init=False)
    availability: IntervalTime = field(default_factory= IntervalTime, init=False)
    extraneous: IntervalTime = field(default_factory= IntervalTime, init=False)

    @cached_property
    def total(self: typing.Self) -> IntervalTime:
        """The total waiting time for an event"""
        return IntervalTime(
            intervals=[
                Interval(
                    begin=self.event.enabled,
                    end=self.event.start,
                ),
            ],
            duration=self.event.start - self.event.enabled,
        )


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
    waiting_time: WaitingTime = field(init=False)
    """The waiting time for the event, split in its components"""
    resource: str | None
    """The resource in charge of the activity"""
    enabled: datetime | None = None
    """The time when the activity was made available for execution"""
    batch: Batch | None = None
    """The batch this event belongs to"""

    def __post_init__(self: typing.Self) -> None:
        self.waiting_time = WaitingTime(self)

    @cached_property
    def execution_time(self: typing.Self) -> timedelta:
        """The execution time elapsed between the event start end the event finalization"""
        return self.end - self.start

    @cached_property
    def total_time(self: typing.Self) -> timedelta:
        """The total event time, between its enablement and the finalization of the execution"""
        return self.waiting_time.total.duration + self.execution_time

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

# type aliases
Log: typing.TypeAlias = typing.Iterable[Event]
Activity: typing.TypeAlias = str
Resource: typing.TypeAlias = str
WeeklyCalendar: typing.TypeAlias = typing.Mapping[int, typing.Iterable[Interval]]
Test: typing.TypeAlias = typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool]

__all__ = ["Batch", "IntervalTime", "WaitingTime", "Event", "Log", "Activity", "Resource", "WeeklyCalendar", "Test"]
