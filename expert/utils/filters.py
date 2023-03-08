import typing
from datetime import timedelta

from expert.model import Event


def always_true(_: Event) -> bool:
    """Return True no matters the event"""
    return True


def always_false(_: Event) -> bool:
    """Return False no matters the event"""
    return False


def has_any_resource(event: Event) -> bool:
    """Return True if the event has a resource assigned"""
    return event.resource is not None


def has_duration(event: Event) -> bool:
    """Return True if the event has a duration greater than 0"""
    return (event.end - event.start).total_seconds() > 0


def has_resource_in(resources: typing.Container[str]) -> typing.Callable[[Event], bool]:
    """Build a filter that returns True if the event's resource is present in the provided collection of resources"""
    return lambda event: event.resource in resources


def has_duration_greater_than(duration: timedelta) -> typing.Callable[[Event], bool]:
    """Build a filter that returns True if the event duration is greater than the provided duration"""
    return lambda event: (event.end - event.start) > duration


def has_duration_greater_or_equal_than(duration: timedelta) -> typing.Callable[[Event], bool]:
    """Build a filter that returns True if the event duration is greater or equal than the provided duration"""
    return lambda event: (event.end - event.start) >= duration


def has_duration_lower_than(duration: timedelta) -> typing.Callable[[Event], bool]:
    """Build a filter that returns True if the event duration is lower than the provided duration"""
    return lambda event: (event.end - event.start) > duration


def has_duration_lower_or_equal_than(duration: timedelta) -> typing.Callable[[Event], bool]:
    """Build a filter that returns True if the event duration is lower or equal than the provided duration"""
    return lambda event: (event.end - event.start) > duration
