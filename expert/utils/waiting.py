from collections import defaultdict
from datetime import timedelta

from intervaltree import Interval, IntervalTree

from expert.model import TimeInterval, Log
from expert.utils.calendars import apply_calendar_to_timeframe, discover_calendars
from expert.utils.log import find_log_end, find_log_start


def decompose_waiting_times(log: Log) -> Log:
    """
    Decompose the waiting time in its separated components

    Parameters
    ----------
    * `log`: the input event log which waiting times will be decomposed

    Returns
    -------
    * the updated log with the event waiting times decomposed
    """
    return __compute_extraneous_times(
        __compute_availability_times(
            __compute_prioritization_times(
                __compute_contention_times(
                    __compute_batching_times(
                        __compute_total_times(log),
                    ),
                ),
            ),
        ),
    )


def __compute_total_times(log: Log) -> Log:
    # Save total waiting time for each event
    for event in log:
        event.waiting_time.total = TimeInterval(
            intervals=[
                Interval(
                    begin=event.enabled,
                    end=event.start,
                ),
            ],
            duration=event.start - event.enabled,
        )

    return log


def __compute_batching_times(log: Log) -> Log:
    # Save batching times for each event
    for event in log:
        # The batching time for an event is the time between it has been enabled and the batch accumulation is done
        if event.batch is None:
            event.waiting_time.batching = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )
        else:
            event.waiting_time.batching = TimeInterval(
                intervals=[
                    Interval(
                        begin=event.enabled,
                        end=event.batch.accumulation.end,
                    ),
                ],
                duration=event.batch.accumulation.end - event.enabled,
            )

    return log


def __compute_contention_times(log: Log) -> Log:
    # build an interval tree with the events for each resource to find overlapping events later
    log_tree = defaultdict(IntervalTree)
    for event in log:
        # only for the events with waiting time
        if event.enabled != event.start:
            log_tree[event.resource][event.enabled:event.end] = event

    # compute the contention time for each event in the log
    for event in log:
        if event.enabled != event.start:
            # get the events that overlap the current one ---i.e., those that overlap the interval [event.enabled: event.start]
            overlapping_events = log_tree[event.resource][event.enabled:event.start]
            # blocking events are the overlapping events that have been both enabled and started executing before the current one
            blocking_events = [
                evt.data for evt in overlapping_events
                if evt.data.enabled < event.enabled and evt.data.start < event.start and evt.data != event
            ]

            if len(blocking_events) > 0:
                blocking_events_tree = IntervalTree()

                # add intervals where blocking events are being executed (trimmed to the start of the event)
                for blocking_evt in blocking_events:
                    blocking_events_tree[blocking_evt.start: min(blocking_evt.end, event.start)] = blocking_evt

                # merge adjacent intervals (adjacent intervals = those separated less than 1 second or overlapping)
                blocking_events_tree.merge_neighbors(
                    distance=timedelta(seconds=1),
                    strict=False,
                )

                # remove intervals covered by the batching time
                for interval in event.waiting_time.batching.intervals:
                    blocking_events_tree.chop(
                        begin=interval.begin,
                        end=interval.end,
                    )

                # collect intervals and aggregate its durations
                event.waiting_time.contention = TimeInterval(
                    intervals=list(blocking_events_tree),
                    duration=sum([interval.end - interval.begin for interval in blocking_events_tree], start=timedelta()),
                )
            else:
                # no contention time
                event.waiting_time.contention = TimeInterval(
                    intervals=[],
                    duration=timedelta(),
                )
        else:
            # no contention time
            event.waiting_time.contention = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )
    return log


def __compute_prioritization_times(log: Log) -> Log:
    # build an interval tree with the events for each resource to find overlapping events later
    log_tree = defaultdict(IntervalTree)
    for event in log:
        if event.enabled != event.start:
            log_tree[event.resource][event.enabled:event.start] = event

    # compute the prioritization time for each event in the log
    for event in log:
        if event.enabled != event.start:
            # blocking events ---i.e., events causing prioritization times--- are those that are enveloped by the current
            # one--- i.e., those that have been enabled after the current one and started executing before it did
            enveloped_events = log_tree[event.resource].envelop(
                begin=event.enabled,
                end=event.start,
            )
            # Ensure the current event is not part of its own blocking events
            blocking_events = [interval.data for interval in enveloped_events if interval.data != event]

            if len(blocking_events) > 0:
                blocking_events_tree = IntervalTree()

                # add intervals where blocking events are being executed
                for blocking_evt in blocking_events:
                    blocking_events_tree[blocking_evt.start: blocking_evt.end] = blocking_evt

                # merge adjacent intervals (adjacent intervals = those separated less than 5 minutes or overlapping)
                blocking_events_tree.merge_neighbors(
                    distance=timedelta(minutes=5),
                    strict=False,
                )

                # remove intervals covered by the batching time
                for interval in event.waiting_time.batching.intervals:
                    blocking_events_tree.chop(
                        begin=interval.begin,
                        end=interval.end,
                    )

                # remove intervals covered by the contention time
                for interval in event.waiting_time.contention.intervals:
                    blocking_events_tree.chop(
                        begin=interval.begin,
                        end=interval.end,
                    )

                # collect intervals and aggregate its durations
                event.waiting_time.prioritization = TimeInterval(
                    intervals=list(blocking_events_tree),
                    duration=sum([interval.end - interval.begin for interval in blocking_events_tree], start=timedelta()),
                )

            else:
                # no prioritization time
                event.waiting_time.prioritization = TimeInterval(
                    intervals=[],
                    duration=timedelta(),
                )
        else:
            # If the event has the same enablement and start timestamps, the waiting time is 0
            event.waiting_time.prioritization = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )
    return log


def __compute_availability_times(log: Log, calendar_granularity: timedelta = timedelta(minutes=60)) -> Log:
    # build an interval for the log timeframe
    log_timeframe = Interval(
        begin=find_log_start(log),
        end=find_log_end(log),
    )
    # compute the availability calendars for the resources
    calendars = discover_calendars(log, calendar_granularity)
    # apply the calendars to the log timeframe
    applied_calendars = {
        resource: apply_calendar_to_timeframe(
            timeframe=log_timeframe,
            weekly_calendar=weekly_calendar,
        ) for (resource, weekly_calendar) in calendars.items()
    }

    # compute the unavailability times for each event in the log
    for event in log:
        # only for the events with waiting time
        if event.enabled != event.start:
            # create a new interval tree with a single interval representing the complete event waiting time
            tree = IntervalTree()
            tree[event.enabled:event.start] = event

            # remove intervals covered by the batching time
            for interval in event.waiting_time.batching.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # remove intervals covered by the contention time
            for interval in event.waiting_time.contention.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # remove intervals covered by the prioritization time
            for interval in event.waiting_time.prioritization.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # remove the intervals where the resource was available
            for interval in applied_calendars[event.resource][event.enabled:event.start]:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # once all availability periods are processed, collect remaining intervals and aggregate its durations
            event.waiting_time.availability = TimeInterval(
                intervals=list(tree),
                duration=sum([interval.end - interval.begin for interval in tree], start=timedelta()),
            )
        else:
            event.waiting_time.availability = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )

    return log


def __compute_extraneous_times(log: Log) -> Log:
    for event in log:
        # only for events with waiting times
        if event.enabled != event.start:
            tree = IntervalTree()

            # initialize the tree with the complete waiting time
            tree[event.enabled:event.start] = event

            # remove intervals covered by the batching time
            for interval in event.waiting_time.batching.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # remove intervals covered by the contention time
            for interval in event.waiting_time.contention.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # remove intervals covered by the prioritization time
            for interval in event.waiting_time.prioritization.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # remove intervals covered by the availability time
            for interval in event.waiting_time.availability.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # collect remaining intervals and aggregate its durations
            event.waiting_time.prioritization = TimeInterval(
                intervals=list(tree),
                duration=sum([interval.end - interval.begin for interval in tree], start=timedelta()),
            )
        else:
            event.waiting_time.prioritization = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )

    return log
