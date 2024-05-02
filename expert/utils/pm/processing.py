from intervaltree import Interval, IntervalTree

from expert.model import Log
from expert.utils.model import TimeInterval
from expert.utils.pm.calendars import discover_calendars
from expert.utils.timer import profile


class ProcessingTimeCanvas:
    """TODO docs"""

    @staticmethod
    def apply(log: Log) -> Log:
        """Decompose the processing time for a log depending on whether the resource was working or not"""
        # build an interval for the log timeframe
        log_timeframe = Interval(
            begin=min([event.start for event in log]),
            end=max([event.end for event in log]),
        )

        # discover and apply the calendars for the resources to the log timeframe
        applied_calendars = {
            resource: calendar.apply(log_timeframe) for (resource, calendar) in discover_calendars(log).items()
        }

        for event in log:
            # only for events with a duration
            if event.start != event.end:
                #################################
                # compute total processing time #
                #################################
                event.processing_time.total = TimeInterval(intervals=[Interval(begin=event.start, end=event.end)])

                ################################
                # compute idle processing time #
                ################################
                processing_time_tree = IntervalTree()
                processing_time_tree[event.start:event.end] = event

                # remove the intervals where the resource is available
                for interval in applied_calendars[event.resource][event.start:event.end]:
                    processing_time_tree.chop(
                        begin=interval.begin,
                        end=interval.end,
                    )
                # once all availability periods are processed, collect remaining intervals
                event.processing_time.idle = TimeInterval(intervals=list(processing_time_tree))

                #####################################
                # compute effective processing time #
                #####################################
                processing_time_tree = IntervalTree()
                processing_time_tree[event.start:event.end] = event
                # remove already explained intervals
                for interval in event.processing_time.idle.intervals:
                    processing_time_tree.chop(interval.begin, interval.end)
                # once all availability periods are processed, collect remaining intervals
                event.processing_time.effective = TimeInterval(intervals=list(processing_time_tree))

        return log
