"""This module provides some utilities for working with logs."""
import typing
from collections import defaultdict
from datetime import datetime, timedelta

from intervaltree import Interval, IntervalTree

from expert.model import Event


def compute_case_duration(case: typing.Iterable[Event]) -> timedelta:
    """
    Compute the duration of a case.

    The duration is computed as the difference between the time the last event ended and the time the first event
    started

    Parameters
    ----------
    * `case`:   *the events from a case*

    Returns
    -------
    * the case duration
    """
    return max(evt.end for evt in case) - min(evt.start for evt in case)

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

def compute_enablement_timestamps(log: typing.Iterable[Event]) -> typing.Iterable[Event]:
    """
    Compute the enablement timestamps for the events in the log.

    This computation is naive, taking as the enablement time for each event the end time of the lastly completed event.

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * the event log with the enablement time updated for the events
    """
    # Create a dict to group the events in cases
    cases: typing.MutableMapping[str, IntervalTree] = defaultdict(IntervalTree)

    log_start = find_log_start(log)

    # Build the timetable for each case
    for event in log:
        cases[event.case].add(Interval(event.start, max(event.end, event.start + timedelta(microseconds=1)), event))

    # For each event, find the last activity from the same case that already finished its execution and set its end time
    # as the enablement timestamp for the current event
    for events in cases.values():
        for begin, _, data in events:
            # Compute the enablement timestamp only if it is not present
            if data.enabled is None:
                previous_events: list[Interval] = sorted(events.envelop(log_start, begin))
                enabler: Event = previous_events[-1].data if len(previous_events) > 0 else None
                # If no event is found before the current one, the enablement time will be this event's start time
                if enabler is not None:
                    data.enabled = min(enabler.end + timedelta(microseconds=1), begin)
                else:
                    data.enabled = begin

    # Flatten the events in the cases dictionary and return them
    return sorted([event.data for tree in cases.values() for event in tree], key = lambda evt: evt.end)
