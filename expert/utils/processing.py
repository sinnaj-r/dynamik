from datetime import timedelta

from intervaltree import Interval, IntervalTree

from expert.model import IntervalTime, Log
from expert.utils.calendars import apply_calendar_to_timeframe, discover_calendars
from expert.utils.log import find_log_end, find_log_start


def decompose_processing_times(log: Log) -> Log:
    """TODO"""
    return __compute_effective_times(
        __compute_idle_times(
            __compute_total_times(log),
        ),
    )


def __compute_total_times(log: Log) -> Log:
    for event in log:
        event.processing_time.total = IntervalTime(
            intervals=[
                Interval(
                    begin=event.start,
                    end=event.end,
                ),
            ],
            duration=event.end - event.start,
        )

    return log


def __compute_idle_times(log: Log, calendar_granularity: timedelta = timedelta(minutes=60)) -> Log:
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
            event.processing_time.idle = IntervalTime(
                intervals=list(tree),
                duration=sum([interval.end - interval.begin for interval in tree], start=timedelta()),
            )
        else:
            event.processing_time.idle = IntervalTime(
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
            event.processing_time.effective = IntervalTime(
                intervals=list(tree),
                duration=sum([interval.end - interval.begin for interval in tree], start=timedelta()),
            )
        else:
            event.processing_time.effective = IntervalTime(
                intervals=[],
                duration=timedelta(),
            )

    return log
