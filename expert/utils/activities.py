"""This module provides some activity-related utilities"""
import typing
from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil

import pandas as pd
from intervaltree import Interval, IntervalTree

from expert.model import Event
from expert.utils.log import compute_batches, find_log_end, find_log_start


def compute_activity_waiting_times(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    """
    Compute the waiting time for each activity in the log.

    To compute the waiting time, events are grouped by their activity and then the waiting times per event are computed.

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * a mapping with pairs (`activity`, [`waiting time`])
    """
    # get the set of activities
    activities = {event.activity for event in log}
    # group events by activity and store the waiting time for each event
    return {
        activity: [event.waiting_time for event in log if event.activity == activity] for activity in sorted(activities)
    }


def compute_activity_execution_times(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    """
    Compute the execution times for each activity in the log.

    To compute the execution time, events are grouped by their activity and then the execution times per event are
    computed.

    Parameters
    ----------
        * `log`:   *an event log*

    Returns
    -------
        * a mapping with pairs (`activity`, [`execution time`])
    """
    # get the set of activities
    activities = {event.activity for event in log}
    # group events by activity and store the waiting time for each event
    return {
        activity: [event.execution_time for event in log if event.activity == activity] for activity in sorted(activities)
    }


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


# compute activity waiting time

#       - batch size???
#   - resource unavailability
#   - extraneous factors


def compute_activity_batching_times(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    """
    Compute the part of the waiting time due to batching accumulation for each activity in the log.

    To compute the batching time, events are grouped by their activity and then the batching times per event are
    computed.

    Parameters
    ----------
        * `log`:   *an event log*

    Returns
    -------
        * a mapping with pairs (`activity`, [`batching time`])
    """
    log = compute_batches(log)
    activities = { event.activity for event in log }

    return { activity: [event.batching_time for event in log if event.activity == activity] for activity in activities}

def compute_activity_contention_times(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    """
    Compute the part of the waiting time due to resource contention for each activity in the log.

    To compute the contention time, we find the blocking events (the ones that share resource and are enabled and began
    execution before the analyzed resource), and compute the contention time as the interval between the first blocking
    event started its execution and the last one finished, or the event started.

    Parameters
    ----------
        * `log`:   *an event log*

    Returns
    -------
        * a mapping with pairs (`activity`, [`contention time`])
    """
    contention_time_per_activity: dict[str, list[timedelta]] = defaultdict(list)

    for event in log:
        blocking_events = [
            evt for evt in log if evt.resource == event.resource and # same resource
                                  # events enabled before the current one that started executing after this was enabled
                                  # but before it started running
                                  evt.enabled < event.enabled < evt.start < event.start
        ]

        if len(blocking_events) > 0:
            contention_time_per_activity[event.activity].append(
                # the contention time is the period between the first blocking event start and the last blocking event end
                # or the current one start, whatever happens first
                min(max(evt.end for evt in blocking_events), event.start) - min(evt.start for evt in blocking_events),
            )
        else:
            # if no blocking events found, the contention is 0
            contention_time_per_activity[event.activity].append(timedelta())
    return contention_time_per_activity

def compute_activity_prioritization_times(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    """
    Compute the part of the waiting time due to activity prioritization for each activity in the log.

    To compute the prioritization time, we find the blocking events (the ones that share resource, are enabled after the
    analyzed event but began executing before it), and compute the prioritization time as the interval between the first
    blocking event started its execution and the last one finished, or the event started.

    Parameters
    ----------
        * `log`:   *an event log*

    Returns
    -------
        * a mapping with pairs (`activity`, [`prioritization time`])
    """
    prioritization_time_per_activity: dict[str, list[timedelta]] = defaultdict(list)

    for event in log:
        blocking_events = [
            evt for evt in log if evt.resource == event.resource and # same resource
                                  # events enabled after this event but that began executing before this one
                                  evt.enabled > event.enabled < evt.start < event.start
        ]

        if len(blocking_events) > 0:
            prioritization_time_per_activity[event.activity].append(
                # the prioritization time is the period between the first blocking event start and the last blocking event
                # end or the current one start, whatever happens first
                min(max(evt.end for evt in blocking_events), event.start) - min(evt.start for evt in blocking_events),
            )
        else:
            # if no blocking events found, the prioritization is 0
            prioritization_time_per_activity[event.activity].append(timedelta())

    return prioritization_time_per_activity
