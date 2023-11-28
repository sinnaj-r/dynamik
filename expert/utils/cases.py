"""This module provides some case-related utilities"""
import typing
from collections import defaultdict
from datetime import timedelta

from expert.model import Log


def compute_inter_arrival_times(log: Log) -> typing.Iterable[timedelta]:
    """TODO DOCS"""
    cases = {}
    # get the start timestamp for each case
    for event in log:
        if event.case not in cases or event.start < cases[event.case]:
            cases[event.case] = event.start

    arrivals = sorted(cases.values())
    # build pairs of start times for successive cases and compute the time elapsed between them
    return sorted([t2-t1 for (t1, t2) in zip(arrivals, arrivals[1:], strict=False)])


def compute_cases_length(log: Log) -> typing.Iterable[int]:
    """TODO DOCS"""
    cases = defaultdict(lambda: 0)

    # count the events in each case
    for event in log:
        cases[event.case] += 1

    return sorted(cases.values())
