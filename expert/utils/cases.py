"""This module provides some case-related utilities"""
import typing
from collections import defaultdict
from datetime import timedelta

from expert.model import Event


def compute_average_case_duration(log: typing.Iterable[Event]) -> timedelta:
    """
    Compute the average case duration for the given log

    Parameters
    ----------
    * `log`:    *the event log*

    Returns
    -------
    * the average case duration
    """
    cases = defaultdict(list)

    for event in log:
        cases[event.case].append(event)

    for key in cases:
        cases[key] = cases[key][-1].end - cases[key][0].start

    return sum(cases.values(), timedelta()) / len(cases)


def compute_average_inter_case_time(log: typing.Iterable[Event], initial_activities: typing.Iterable[str]) -> timedelta:
    """
    Compute the average time between new cases

    Parameters
    ----------
    * `log`:                *the event log*
    * `initial_activities`: *the list of initial activities*

    Returns
    -------
    * the average time between cases
    """
    # get the start timestamps
    starts = sorted([event.start for event in log if event.activity in initial_activities])
    # compute the time between successive starts
    distances = [event2 - event1 for (event1, event2) in zip(starts[:-1], starts[1:], strict=True)]
    return sum(distances, timedelta()) / len(distances)


