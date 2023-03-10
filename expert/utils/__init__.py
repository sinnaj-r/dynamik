"""This module provides some utilities for working with logs."""
import logging
import typing
from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil

import numpy as np
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
    return sorted([event.data for tree in cases_timetable.values() for event in tree], key = lambda evt: evt.end)

def compute_activity_arrival_rate(
        log: typing.Iterable[Event],
        timeunit: timedelta = timedelta(minutes=1),
) -> typing.Mapping[str, float]:
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

    activity_rates = {}
    # Compute activity arrival rates
    for activity in activities:
        events_per_slot = []
        for slot in slots:
            # Get the events with the given activity that started in the time slot
            events_in_slot = len(event_timetable[activity][slot.begin:slot.end])

            events_per_slot.append(events_in_slot)
            logging.debug("activity %(activity)s slot %(slot)r events %(events)d",
                          {"activity": activity, "slot": slot, "events": events_in_slot})

        # Summarize the results for every activity computing the average arrival rate per time slot
        activity_rates[activity] = float(np.mean(np.array(events_per_slot)))

    return {key: activity_rates[key] for key in sorted(activity_rates.keys())}


def compute_resources_utilization_rate(
        log: typing.Iterable[Event],
        timeunit: timedelta = timedelta(minutes=1),
) -> typing.Mapping[str, float]:
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
    # Filter events without resource
    filtered_log: list[Event] = [event for event in log if has_any_resource(event)]
    # Get the set of resources
    resources: set[str] = {event.resource for event in filtered_log}

    # Build time slots where frequency will be checked
    log_start: datetime = find_log_start(log)
    log_end: datetime = find_log_end(log)

    intervals = pd.interval_range(start=log_start, periods=ceil((log_end - log_start) / timeunit), freq=timeunit).array
    slots: list[Interval] = [Interval(interval.left, interval.right) for interval in intervals]

    # Build resources timetables
    resources_timetable: defaultdict[str, IntervalTree] = defaultdict(IntervalTree)
    for event in filtered_log:
        resources_timetable[event.resource][event.start:(event.end+timedelta(microseconds=1))] = event

    # Compute resource occupancy for each resource
    resources_occupancy: defaultdict[str, list[float]] = defaultdict(list)
    for resource in resources:
        for slot in slots:
            # Get the events in the time slot for the resource
            events_in_slot = resources_timetable[resource][slot.begin:slot.end]
            # Compute the percentage of the slot time that is used by the events
            used_time_in_slot = sum(
                ((min(event.end, slot.end) - max(event.begin, slot.begin)) for event in events_in_slot),
                start=timedelta(),
            ) / (slot.end - slot.begin)
            resources_occupancy[resource].append(used_time_in_slot)
            logging.debug("resource %(resource)s slot %(slot)r usage %(usage)06f",
                          {"resource": resource, "slot": slot, "usage": used_time_in_slot})

    # Summarize the results computing the average resource usage for every time slot
    return {
        key: float(np.mean(np.array(resources_occupancy[key])))
        for key in sorted(resources_occupancy.keys())
    }



def compute_activity_waiting_times(
        log: typing.Iterable[Event],
) -> typing.Mapping[str, timedelta]:
    """Compute the average waiting time for each activity in the log.

    To compute the average waiting time events are grouped by their activity and then the waiting times per event are
    aggregated by the mean.

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * a mapping with pairs (`activity`, `average waiting time`)
    """
    # get the set of activities
    activities = {event.activity for event in log}
    # group events by activity and store the waiting time for each event
    waiting_times = {
        activity: [event.waiting_time for event in log if event.activity == activity] for activity in activities
    }
    # aggregate events waiting time by the average
    return {
        activity: sum(waiting_times[activity], timedelta(0)) / len(waiting_times[activity]) for activity in activities
    }


















