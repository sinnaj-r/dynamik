"""This module provides some log-related utilities"""
import typing
from datetime import datetime

from expert.model import Event


def find_log_start(log: typing.Iterable[Event]) -> datetime:
    """
    Find the timestamp when executions in the log began (i.e., the minimum start time from the events)

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * the start timestamp for the log, computed as the minimum start timestamp from the events
    """
    return min(event.start for event in log)


def find_log_end(log: typing.Iterable[Event]) -> datetime:
    """
    Find the timestamp when executions in the log ended (i.e., the maximum end time from the events)

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * the end timestamp for the log, computed as the maximum end timestamp from the events
    """
    return max(event.end for event in log)


def infer_initial_activities(log: typing.Iterable[Event]) -> typing.Iterable[str]:
    """
    Infer the start activities of a process from the log execution.

    Parameters
    ----------
    * `log`:    *the event log*

    Returns
    -------
    * the set of activities that are seen first from the cases
    """
    log = list(log)
    first_event_per_case = {}

    for event in log:
        if event.case not in first_event_per_case or event.start < first_event_per_case[event.case].start:
            first_event_per_case[event.case] = event

    return {event.activity for event in first_event_per_case.values()}


def infer_final_activities(log: typing.Iterable[Event]) -> typing.Iterable[str]:
    """
    Infer the end activities of a process from the log execution.

    Parameters
    ----------
    * `log`:    *the event log*

    Returns
    -------
    * the set of activities that are seen last for each case
    """
    log = list(log)
    last_event_per_case = {}

    for event in log:
        if event.case not in last_event_per_case or event.end > last_event_per_case[event.case].end:
            last_event_per_case[event.case] = event

    return {event.activity for event in last_event_per_case.values()}
