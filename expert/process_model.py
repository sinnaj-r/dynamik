"""This module contains the definition of a model for the events from an event log."""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property

from intervaltree import Interval

from expert.utils.model import TimeInterval


@dataclass
class Batch:
    """A batch descriptor"""

    activity: str
    """The activity performed during the batch"""
    resource: str
    """The resource executing the batch"""
    events: typing.Iterable[Event]
    """The events executed in this batch"""

    @cached_property
    def size(self: typing.Self) -> int:
        """The batch size"""
        return len(list(self.events))

    @cached_property
    def accumulation(self: typing.Self) -> Interval[datetime]:
        """The batch accumulation interval"""
        return Interval(
            # the first enabled event
            begin=min(event.enabled for event in self.events),
            # the last enabled event
            end=max(event.enabled for event in self.events),
        )

    @cached_property
    def execution(self: typing.Self) -> Interval[datetime]:
        """The batch execution interval"""
        return Interval(
            # the first started event
            begin=min(event.start for event in self.events),
            # the last ended event
            end=max(event.end for event in self.events),
        )

    def asdict(self: typing.Self) -> dict:
        return {
            "activity": self.activity,
            "resource": self.resource,
            "events": [
                {
                    "case": event.case,
                    "activity": event.activity,
                    "resource": event.resource,
                    "start": {
                        "year": event.start.year,
                        "month": event.start.month,
                        "day": event.start.day,
                        "hour": event.start.hour,
                        "minute": event.start.minute,
                        "second": event.start.second,
                        "microsecond": event.start.microsecond,
                    },
                    "end": {
                        "year": event.end.year,
                        "month": event.end.month,
                        "day": event.end.day,
                        "hour": event.end.hour,
                        "minute": event.end.minute,
                        "second": event.end.second,
                        "microsecond": event.end.microsecond,
                    },
                    "enabled": {
                        "year": event.enabled.year,
                        "month": event.enabled.month,
                        "day": event.enabled.day,
                        "hour": event.enabled.hour,
                        "minute": event.enabled.minute,
                        "second": event.enabled.second,
                        "microsecond": event.enabled.microsecond,
                    },
                } for event in self.events
            ],
        }


@dataclass
class WaitingTime:
    """A representation of the waiting time for an event, with its decomposition"""

    batching: TimeInterval = field(default_factory=TimeInterval)
    contention: TimeInterval = field(default_factory=TimeInterval)
    prioritization: TimeInterval = field(default_factory=TimeInterval)
    availability: TimeInterval = field(default_factory=TimeInterval)
    extraneous: TimeInterval = field(default_factory=TimeInterval)
    total: TimeInterval = field(default_factory=TimeInterval)

    def asdict(self: typing.Self) -> dict:
        return {
            "batching": self.batching.asdict(),
            "contention": self.contention.asdict(),
            "prioritization": self.prioritization.asdict(),
            "availability": self.availability.asdict(),
            "extraneous": self.extraneous.asdict(),
            "total": self.total.asdict(),
        }


@dataclass
class ProcessingTime:
    """An object representing the processing time for an event, with its decomposition"""

    effective: TimeInterval = field(default_factory=TimeInterval)
    idle: TimeInterval = field(default_factory=TimeInterval)
    total: TimeInterval = field(default_factory=TimeInterval)

    def asdict(self: typing.Self) -> dict:
        return {
            "effective": self.effective.asdict(),
            "idle": self.idle.asdict(),
            "total": self.total.asdict(),
        }


@dataclass(slots=True)
class Event:
    """
    `Event` provides a standardized representation of an event from a log.

    An event in from a process log represents the execution of an activity in a specific instant in time by a specific
    resource in the context of a specific process execution.
    """

    case: str
    """The case identifier for the event, which associates it with a specific process execution"""
    activity: Activity
    """The activity being executed"""
    resource: Resource | None
    """The resource in charge of the activity"""
    start: datetime
    """The time when the activity execution began"""
    end: datetime
    """The time when the activity execution ended"""
    enabled: datetime | None = None
    """The time when the activity was made available for execution"""
    batch: Batch | None = field(default=None, hash=False, compare=False, repr=False)
    """The batch this event belongs to"""
    waiting_time: WaitingTime = field(default_factory=WaitingTime, hash=False, compare=False, repr=False)
    """The waiting time for the event, split in its components"""
    processing_time: ProcessingTime = field(default_factory=ProcessingTime, hash=False, compare=False, repr=False)
    """The processing time for the event, split in its components"""
    attributes: typing.Mapping[str, typing.Any] = field(default_factory=dict, hash=False, compare=False)
    """The additional attributes for the event"""

    @property
    def cycle_time(self: typing.Self) -> timedelta:
        """The total time for the event"""
        return self.end - self.enabled

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

    def __hash__(self: typing.Self) -> int:
        return hash((self.case, self.activity, self.resource, self.start, self.end, self.enabled))

    def asdict(self: typing.Self) -> dict:
        """Convert the object to a dict"""
        result = {
            "case": self.case,
            "activity": self.activity,
            "resource": self.resource,
            "start": {
                "year": self.start.year,
                "month": self.start.month,
                "day": self.start.day,
                "hour": self.start.hour,
                "minute": self.start.minute,
                "second": self.start.second,
                "microsecond": self.start.microsecond,
            },
            "end": {
                "year": self.end.year,
                "month": self.end.month,
                "day": self.end.day,
                "hour": self.end.hour,
                "minute": self.end.minute,
                "second": self.end.second,
                "microsecond": self.end.microsecond,
            },
            "enabled": {
                "year": self.enabled.year,
                "month": self.enabled.month,
                "day": self.enabled.day,
                "hour": self.enabled.hour,
                "minute": self.enabled.minute,
                "second": self.enabled.second,
                "microsecond": self.enabled.microsecond,
            },
            "batch": self.batch.asdict(),
            "waiting_time": self.waiting_time.asdict(),
            "processing_time": self.processing_time.asdict(),
            "attributes": self.activity,
        }

        return result


# type aliases
Log: typing.TypeAlias = typing.Iterable[Event]
Trace: typing.TypeAlias = typing.Iterable[Event]
Activity: typing.TypeAlias = str
Resource: typing.TypeAlias = str
