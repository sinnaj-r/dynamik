from datetime import timedelta

from intervaltree import Interval, IntervalTree

from expert.process_model import Log
from expert.utils.model import TimeInterval
from expert.utils.pm.calendars import discover_calendars


def decompose_processing_times(log: Log) -> Log:
    """TODO"""
    return __compute_effective_times(
        __compute_idle_times(
            __compute_total_times(log),
        ),
    )


def __compute_total_times(log: Log) -> Log:
    for event in log:
        event.processing_time.total = TimeInterval(
            intervals=[
                Interval(
                    begin=event.start,
                    end=event.end,
                ),
            ],
            duration=event.end - event.start,
        )

    return log


def __compute_idle_times(log: Log) -> Log:
    # build an interval for the log timeframe
    log_timeframe = Interval(
        begin=min(event.start for event in log),
        end=max(event.end for event in log),
    )
    # compute the availability calendars for the resources
    calendars = discover_calendars(log)
    # apply the calendars to the log timeframe
    applied_calendars = {
        resource: calendar.apply(log_timeframe) for (resource, calendar) in calendars.items()
    }

    # compute the unavailability times for each event in the log
    for event in log:
        # only for events with a duration
        if event.start != event.end:
            # create a new interval tree with a single interval representing the complete event processing time
            tree = IntervalTree()
            tree[event.start:event.end] = event

            # remove the intervals where the resource is available
            for interval in applied_calendars[event.resource][event.start:event.end]:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # once all availability periods are processed, collect remaining intervals and aggregate its durations
            event.processing_time.idle = TimeInterval(
                intervals=list(tree),
                duration=sum([interval.end - interval.begin for interval in tree], start=timedelta()),
            )
        else:
            event.processing_time.idle = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )

    return log


def __compute_effective_times(log: Log) -> Log:
    for event in log:
        # only for events with a duration
        if event.start != event.end:
            # initialize the tree with the complete processing time
            tree = IntervalTree()
            tree[event.start:event.end] = event

            # remove intervals covered by idle time
            for interval in event.processing_time.idle.intervals:
                tree.chop(
                    begin=interval.begin,
                    end=interval.end,
                )

            # collect remaining intervals and aggregate its durations
            event.processing_time.effective = TimeInterval(
                intervals=list(tree),
                duration=sum([interval.end - interval.begin for interval in tree], start=timedelta()),
            )
        else:
            event.processing_time.effective = TimeInterval(
                intervals=[],
                duration=timedelta(),
            )

    return log
