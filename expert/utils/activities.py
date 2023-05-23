"""This module provides some activity-related utilities"""
import typing
from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil

import pandas as pd
from intervaltree import Interval, IntervalTree

from expert.model import Event
from expert.utils.log import compute_batches, find_log_end, find_log_start


def compute_activity_arrival_rate(
        log: typing.Iterable[Event],
        timeunit: timedelta = timedelta(minutes=1),
) -> typing.Mapping[str, typing.Iterable[float]]:
    """Compute the arrival rate for each activity in the log.

    To compute the arrival rate, the log timeframe is split in slots of size `timeunit` and, for each of these slots,
    we count how many events started within the slot timeframe. Finally, the count of events per slot is reduced by a
    mean, so we have the mean arrival rate per activity.

    Parameters
    ----------
    * `log`:        *an event log*
    * `timeunit`:   *the granularity used to compute the arrival rate*

    Returns
    -------
    * a mapping with pairs (`activity`, `arrival rate`)
    """
    # Get the set of activities
    activities = {event.activity for event in log}
    # Build time slots where arrival rate will be checked
    log_start: datetime = find_log_start(log)
    log_end: datetime = find_log_end(log)
    intervals = pd.interval_range(start=log_start, periods=ceil((log_end - log_start) / timeunit), freq=timeunit).array
    slots: list[Interval] = [Interval(interval.left, interval.right) for interval in intervals]
    # Build a timetable with all the event start timestamps for each activity
    event_timetable = defaultdict(IntervalTree)
    for event in log:
        event_timetable[event.activity][event.start:(event.start + timedelta(microseconds=1))] = event

    # Compute arrival rates for each activity and time slot
    return {
        activity: [ len(event_timetable[activity][slot.begin:slot.end]) for slot in slots ]
        for activity in sorted(activities)
    }


def compute_activity_batch_sizing(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    """
    Compute the size of the batches for each activity in the log.

    Parameters
    ----------
        * `log`:   *an event log*

    Returns
    -------
        * a mapping with pairs (`activity`, [`batch size`])
    """
    if any(event.batch is None for event in log):
        log = compute_batches(log)

    # get the set of batches per activity
    batch_per_activity = defaultdict(set)
    for event in log:
        batch_per_activity[event.activity].add(event.batch)

    # get the batch sizes for each activity
    return { activity: [batch.size for batch in batches] for (activity, batches) in batch_per_activity }


def compute_prioritized_activities(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, set[str]]:
    """
    Compute the prioritized activities for each activity in the log.

    To compute the prioritized activities, we find the blocking events (the ones that share resource, are enabled  after
    the analyzed event but began executing before it), store the event activity as prioritized over the current one.

    Parameters
    ----------
        * `log`:   *an event log*

    Returns
    -------
        * a mapping with pairs (`activity`, [`prioritized activities`])
    """
    prioritized_activities_per_activity: dict[str, set[str]] = defaultdict(set)

    for event in log:
        # save the activities that are prioritized over the current one
        prioritized_activities_per_activity[event.activity] = prioritized_activities_per_activity[event.activity].union(
            {
                evt.activity for evt in log
                if evt.resource == event.resource and evt.enabled > event.enabled < evt.start < event.start
            },
        )

    return prioritized_activities_per_activity


def compute_activity_resources(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, set[str]]:
    """
    Compute the resource allocations for each activity

    Parameters
    ----------
    * `log`:        *an event log*

    Returns
    -------
    * a mapping with pairs (`activity`, {`resources`}) where resources is the list of resources executing the activity
    """
    # Filter out the events without assigned resources
    filtered_log: list[Event] = [ event for event in log if event.resource is not None ]
    # Extract the set of activities from the event log
    activities: set[str] = { event.activity for event in filtered_log }

    # create the allocations map
    allocations = defaultdict(set)
    # populate the allocations map
    for activity in activities:
        allocations[activity] = { event.resource for event in filtered_log if event.activity == activity }

    # Get the set of resources executing each activity
    return allocations
