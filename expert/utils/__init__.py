"""This module provides some utilities for working with logs."""
import typing
from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil

import pandas as pd
from intervaltree import Interval, IntervalTree

from expert.model import Event
from expert.utils.filters import has_any_resource


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
    # Convert the log to a list (to prevent problems with generators and reiterating over the log)
    log=list(log)

    # Create a dict to group the events in cases
    cases_timetable: typing.MutableMapping[str, IntervalTree] = defaultdict(IntervalTree)

    # Compute the log start time
    log_start = find_log_start(log)

    # Build the timetable for each case
    for event in log:
        cases_timetable[event.case].add(Interval(event.start, max(event.end, event.start + timedelta(microseconds=1)), event))

    # For each event, find the last activity from the same case that already finished its execution and set its end time
    # as the enablement timestamp for the current event
    for events in cases_timetable.values():
        for begin, _, data in events:
            previous_events: list[Interval] = sorted(events.envelop(log_start, begin))
            enabler: Event = previous_events[-1].data if len(previous_events) > 0 else None
            # If no event is found before the current one, the enablement time will be this event's start time
            if enabler is not None:
                data.enabled = min(enabler.end + timedelta(microseconds=1), begin)
            else:
                data.enabled = begin

    # Flatten the events in the cases dictionary and return them
    return sorted([event.data for tree in cases_timetable.values() for event in tree], key = lambda evt: (evt.end, evt.start))


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


def compute_resources_utilization_rate(
        log: typing.Iterable[Event],
        timeunit: timedelta = timedelta(minutes=1),
) -> typing.Mapping[str, typing.Iterable[float]]:
    """Compute the mean resource utilization for each resource in the event log.

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
    filtered_log: list[Event] = [event for event in log if has_any_resource(event)]
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



# compute activity duration

# compute activity waiting time
#   - activity batching time
#       - batch accumulation
#       - batch ready
#       - batch size???
#   - resource contention
#   - resource unavailability
#   - activity prioritization
#   - extraneous factors


def compute_activity_batching_time(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, typing.Iterable[timedelta]]:
    ...







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
    cases = {}

    for event in log:
        if event.case not in cases:
            cases[event.case] = event.activity

    return set(cases.values())


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
    cases = {}

    for event in log:
        cases[event.case] = event.activity

    return set(cases.values())


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
    times = []
    last = None

    for event in log:
        if event.activity in initial_activities and last is None:
            last = event
        elif event.activity in initial_activities:
            times.append(event.start - last.end)
            last = event

    return sum(times, timedelta()) / len(times)






