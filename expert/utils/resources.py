"""This module provides some resource-related utilities"""
import typing
from collections import defaultdict
from datetime import datetime, time, timedelta
from math import ceil

import pandas as pd
from intervaltree import Interval, IntervalTree

from expert.model import Event
from expert.utils.log import find_log_end, find_log_start


def compute_resources_utilization_rate(
        log: typing.Iterable[Event],
        timeunit: timedelta = timedelta(minutes=1),
) -> typing.Mapping[str, typing.Iterable[float]]:
    """
    Compute the mean resource utilization for each resource in the event log.

    To perform this computation, the log timeframe is split in slots of size `timeunit` and, for each of these slots,
    the rate of occupation is computed by intersecting it with the events executed by each resource. Finally, the mean
    utilization rate for each resource is computed with the mean of the resource utilization for every time slot.

    Parameters
    ----------
    * `log`:        *an event log*
    * `timeunit`:   *the granularity used to compute the resources utilization rate

    Returns
    -------
    * a mapping with pairs (`resource`, `utilization`) where usage is in the range [0.0, 1.0], being 0.0 no activity
      at all and 1.0 a fully used resource
    """
    # Filter out the events without assigned resources
    filtered_log: list[Event] = [event for event in log if event.resource is not None]
    # Extract the set of resources from the event log
    resources: set[str] = {event.resource for event in filtered_log}
    # Build the time slots where the frequency will be checked
    log_start: datetime = find_log_start(log)
    log_end: datetime = find_log_end(log)
    intervals = pd.interval_range(start=log_start, periods=ceil((log_end - log_start) / timeunit), freq=timeunit).array
    slots: list[Interval] = [Interval(interval.left, interval.right) for interval in intervals]
    # Build the event execution timetable for each resource
    resources_timetable: defaultdict[str, IntervalTree] = defaultdict(IntervalTree)
    for event in filtered_log:
        resources_timetable[event.resource][event.start:event.end] = event
    # Compute resource occupancy for each resource
    resources_occupancy: dict[str, list[float]] = defaultdict(list)
    for resource in sorted(resources):
        for slot in slots:
            # Get the events in the time slot for the resource
            events_in_slot = resources_timetable[resource][slot.begin:slot.end]
            # Compute the percentage of the slot time that is used by the events
            used_time_in_slot = sum(
                ((min(event.end, slot.end) - max(event.begin, slot.begin)) for event in events_in_slot),
                start=timedelta(),
            ) / (slot.end - slot.begin)
            resources_occupancy[resource].append(used_time_in_slot)

    return resources_occupancy


def compute_resources_availability_calendars(
        log: typing.Iterable[Event],
        granularity: timedelta=timedelta(minutes=1),
) -> typing.Mapping[str, typing.Mapping[int, typing.Mapping[tuple[time, time], bool]]]:
    """
    Compute the availability calendar for each resource in the log.

    Parameters
    ----------
    * `log`:            *an event log*
    * `granularity`:    *the granularity used to compute the resources utilization rate

    Returns
    -------
    * a mapping with pairs (`resource`, `calendar`) where calendar is an iterable of booleans indicating if the resource
    was available at the interval i for intervals of size granularity
    """
    # build a map with the events executed by each resource
    events_per_resource = defaultdict(list)
    for event in log:
        events_per_resource[event.resource].append(event)

    # compute the calendar for each resource
    calendars = {}
    for (resource, events) in events_per_resource.items():
        calendars[resource] = __compute_calendar(events, granularity)

    return calendars


def __compute_calendar(
        log: typing.Iterable[Event],
        granularity: timedelta=timedelta(minutes=1),
) -> typing.Mapping[int, typing.Mapping[tuple[time, time], bool]]:
    calendar: dict[int, dict[tuple[time, time], bool]] = defaultdict(dict)

    for event in log:
        # store the event start and end timestamps
        start = event.start.time()
        end = event.end.time()
        # store the start and end weekdays
        start_weekday = event.start.weekday()
        end_weekday = event.end.weekday()
        # compute the start and end slots for given granularity
        start_slot_idx = int((start.second + start.minute * 60 + start.hour * 60 * 60) // granularity.total_seconds())
        end_slot_idx = int((end.second + end.minute * 60 + end.hour * 60 * 60) // granularity.total_seconds())

        start_slot: tuple[time, time] = (
            time(
                hour=int(((start_slot_idx * granularity.total_seconds()) // 3600) % 24),
                minute=int(((start_slot_idx * granularity.total_seconds()) // 60) % 60),
                second=int((start_slot_idx * granularity.total_seconds()) % 60),
            ),
            time(
                hour=int(((start_slot_idx * granularity.total_seconds() + granularity.total_seconds()) // 3600) % 24),
                minute=int(((start_slot_idx * granularity.total_seconds() + granularity.total_seconds()) // 60) % 60),
                second=int((start_slot_idx * granularity.total_seconds() + granularity.total_seconds()) % 60),
            ),
        )

        end_slot: tuple[time, time] = (
            time(
                hour=int(((end_slot_idx * granularity.total_seconds()) // 3600) % 24),
                minute=int(((end_slot_idx * granularity.total_seconds()) // 60) % 60),
                second=int((end_slot_idx * granularity.total_seconds()) % 60),
            ),
            time(
                hour=int(((end_slot_idx * granularity.total_seconds() + granularity.total_seconds()) // 3600) % 24),
                minute=int(((end_slot_idx * granularity.total_seconds() + granularity.total_seconds()) // 60) % 60),
                second=int((end_slot_idx * granularity.total_seconds() + granularity.total_seconds()) % 60),
            ),
        )

        # set the slots for the event start and end as active
        calendar[start_weekday][start_slot] = True
        calendar[end_weekday][end_slot] = True

    return calendar
