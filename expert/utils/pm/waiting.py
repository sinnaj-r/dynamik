from collections import defaultdict
from datetime import timedelta

from intervaltree import Interval, IntervalTree

from expert.model import Log
from expert.utils.model import TimeInterval
from expert.utils.pm.calendars import discover_calendars


class WaitingTimeCanvas:
    """TODO docs"""

    @staticmethod
    def apply(log: Log) -> Log:
        """Decompose the waiting times from given log applying the waiting time canvas."""
        # build an intervaltree with the log for each resource
        busy_resource_tree = defaultdict(IntervalTree)
        for event in log:
            busy_resource_tree[event.resource][event.start:event.end] = event

        # build an interval for the log timeframe
        log_timeframe = Interval(
            begin=min(event.start for event in log),
            end=max(event.end for event in log),
        )

        # discover and apply the calendars for the resources to the log timeframe
        applied_calendars = {
            resource: calendar.apply(log_timeframe) for (resource, calendar) in discover_calendars(log).items()
        }

        # compute the waiting times for each event
        for event in log:
            # create a new list to keep track of the already explained intervals
            already_explained = []

            # compute only for events that have a waiting time
            if event.enabled != event.start:
                ##################################
                # compute the total waiting time #
                ##################################
                event.waiting_time.total = TimeInterval(
                    intervals=[
                        Interval(
                            begin=event.enabled,
                            end=event.start,
                        ),
                    ],
                )
                #############################
                # compute the batching time #
                #############################
                if event.batch is not None:
                    event.waiting_time.batching = TimeInterval(
                        intervals=[
                            # The batching time for an event is the interval between it has been enabled and the batch accumulation is done
                            Interval(
                                begin=event.enabled,
                                end=event.batch.accumulation.end,
                            ),
                        ],
                    )
                # store batching intervals as already explained
                already_explained.extend(event.waiting_time.batching.intervals)

                # get the events that overlap the current one ---i.e., those that overlap the interval [event.enabled: event.start]
                overlapping_events = [interval.data for interval in busy_resource_tree[event.resource][event.enabled:event.start]]

                ############################
                # compute contention times #
                ############################
                contention_tree: IntervalTree = IntervalTree()
                for evt in overlapping_events:
                    if evt.enabled < event.enabled and evt != event:
                        # evt causes contention between its start and its end or the next event starts
                        contention_tree[evt.start: min(evt.end, event.start)] = evt
                # remove already explained waiting intervals
                for interval in already_explained:
                    contention_tree.chop(interval.begin, interval.end)
                # merge adjacent intervals in contention tree
                contention_tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)
                # collect contention intervals
                event.waiting_time.contention = TimeInterval(intervals=list(contention_tree))
                # store contention intervals as already explained
                already_explained.extend(event.waiting_time.contention.intervals)

                ################################
                # compute prioritization times #
                ################################
                prioritization_tree: IntervalTree = IntervalTree()
                for evt in overlapping_events:
                    if evt.enabled > event.enabled and evt != event:
                        # event causes prioritization between its start and its end or the next event starts
                        prioritization_tree[evt.start: min(evt.end, event.start)] = evt
                # remove already explained waiting intervals
                for interval in already_explained:
                    prioritization_tree.chop(interval.begin, interval.end)
                # merge adjacent intervals in prioritization tree
                prioritization_tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)
                # collect contention intervals
                event.waiting_time.prioritization = TimeInterval(intervals=list(prioritization_tree))
                # store prioritization intervals as already explained
                already_explained.extend(event.waiting_time.prioritization.intervals)

                ##################################
                # compute the availability times #
                ##################################
                # create a new interval tree with a single interval representing the complete event waiting time
                unavailability_tree = IntervalTree([Interval(begin=event.enabled, end=event.start)])
                # remove the intervals where the resource was available
                for interval in applied_calendars[event.resource][event.enabled:event.start]:
                    unavailability_tree.chop(
                        begin=interval.begin,
                        end=interval.end,
                    )
                # remove already explained waiting intervals
                for interval in already_explained:
                    unavailability_tree.chop(interval.begin, interval.end)
                # merge adjacent intervals in availability tree
                unavailability_tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)
                # collect unavailability intervals
                event.waiting_time.availability = TimeInterval(intervals=list(unavailability_tree))
                # store prioritization intervals as already explained
                already_explained.extend(event.waiting_time.availability.intervals)

                ############################
                # compute extraneous times #
                ############################
                # create an intervaltree with the full waiting time
                extraneous_tree = IntervalTree([Interval(begin=event.enabled, end=event.start)])
                # remove the already explained intervals
                for interval in already_explained:
                    extraneous_tree.chop(interval.begin, interval.end)
                # merge adjacent intervals in extraneous tree
                extraneous_tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)
                # collect unavailability intervals
                event.waiting_time.extraneous = TimeInterval(intervals=list(extraneous_tree))

        return log
