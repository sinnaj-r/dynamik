"""This module provides some log-related utilities"""
import typing
from collections import defaultdict
from datetime import datetime, timedelta

from intervaltree import Interval, IntervalTree

from expert.model import Batch, Event


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


def compute_batches(log: typing.Iterable[Event]) -> typing.Iterable[Event]:
    """
    Compute the batches in the event log and add their descriptor to the events.

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * the event log with the batches information
    """
    batches = []

    # Group events by resource and activity
    events_per_resource_and_activity = defaultdict(lambda: defaultdict(list))
    for event in log:
        events_per_resource_and_activity[event.resource][event.activity].append(event)

    # Build batches for each resource and activity
    for events_per_resource in events_per_resource_and_activity.values():
        for events_per_activity in events_per_resource.values():
            # Create a new empty batch (new activity implies new batch)
            current_batch = []

            for event in sorted(events_per_activity, key=lambda evt: evt.enabled):
                # Add the event to the current batch if it was enabled in time or in first iteration
                if (len(current_batch) == 0 or min(evt.enabled for evt in current_batch) <=
                        event.enabled <= min(evt.start for evt in current_batch)):
                    current_batch.append(event)
                # If the event is not part of the current batch, save the current batch and create a new one with the
                # current event
                else:
                    batches.append(current_batch)
                    current_batch = [event]

            # Save the batch for the last iteration
            batches.append(current_batch)

    # Build batch descriptors and add them to the events
    for batch in batches:
        batch_descriptor = Batch(
            activity=batch[0].activity,
            resource=batch[0].resource,
            accumulation=Interval(
                # the first enabled event
                begin=min(event.enabled for event in batch),
                # the last enabled event
                end=max(event.enabled for event in batch),
            ),
            execution=Interval(
                # the first started event
                begin=min(event.start for event in batch),
                # the last ended event
                end=max(event.end for event in batch),
            ),
            # the batch size
            size=len(batch),
        )

        for event in batch:
            event.batch = batch_descriptor

    return log


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
    log=list(log)
    cases = {}

    for event in sorted(log, key=lambda evt: evt.start):
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
    log = list(log)
    cases = {}

    for event in sorted(log, key=lambda evt: evt.end):
        cases[event.case] = event.activity

    return set(cases.values())
