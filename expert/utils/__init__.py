"""
This module provides you with some utilities like event filters that can be used to preprocess the log and discard
some events and functions to find the beginning or end of an event log and to compute the enablement timestamps if they
are not present.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from intervaltree import Interval, IntervalTree

from expert.model import Event


def compute_case_duration(case: list[Event]) -> timedelta:
    """
    Compute the duration of a case as the difference between the time the last event ended and the time the first event
    started
    """
    return max(evt.end for evt in case) - min(evt.start for evt in case)

def find_log_start(log: Iterable[Event]) -> datetime:
    """TODO"""
    return min(event.start for event in log)


def find_log_end(log: Iterable[Event]) -> datetime:
    """TODO"""
    return max(event.end for event in log)

def compute_enablement_timestamps(log: Iterable[Event]) -> Iterable[Event]:
    """TODO"""
    # Create a dict to group the events in cases
    cases: defaultdict[str, IntervalTree] = defaultdict(IntervalTree)

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
