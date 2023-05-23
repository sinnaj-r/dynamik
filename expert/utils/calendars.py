import typing
from collections import defaultdict
from datetime import time, timedelta

from intervaltree import Interval, IntervalTree

from expert.model import Log, Resource, WeeklyCalendar


def __compute_calendar(
        log: Log,
        granularity: timedelta=timedelta(minutes=15),
) -> WeeklyCalendar:
    # create a new empty weekly calendar
    calendar: dict[int, set] = defaultdict(set)

    # available slots are those where start/end are present ---i.e., the resource performed some activity---.
    for event in log:
        start = event.start.time()
        # the slot index is the seconds count since midnight, rounded to the selected granularity
        start_slot_index = int((start.second + start.minute * 60 + start.hour * 60 * 60) // granularity.total_seconds())
        start_slot = Interval(
            # the slot starts at index * granularity
            begin= time(
                hour=int(((start_slot_index * granularity.total_seconds()) // 3600) % 24),
                minute=int(((start_slot_index * granularity.total_seconds()) // 60) % 60),
                second=int((start_slot_index * granularity.total_seconds()) % 60),
            ),
            # the slot ends at index * granularity + granularity
            # subtracting 1 second we ensure we get "correct" intervals and prevent the case where (23:45 > 00:00)
            end=time(
                hour=int(((start_slot_index*granularity.total_seconds()+granularity.total_seconds()-1)//3600)%24),
                minute=int(((start_slot_index*granularity.total_seconds()+granularity.total_seconds()-1)//60)%60),
                second=int((start_slot_index*granularity.total_seconds()+granularity.total_seconds()-1)%60),
            ),
        )
        # add the slot to the resource calendar for the corresponding weekday
        calendar[event.start.weekday()].add(start_slot)

        # repeat the same procedure as before, but for the ending time of the event
        end = event.end.time()
        end_slot_index = int((end.second + end.minute * 60 + end.hour * 60 * 60) // granularity.total_seconds())
        end_slot = Interval(
            begin=time(
                hour=int(((end_slot_index * granularity.total_seconds()) // 3600) % 24),
                minute=int(((end_slot_index * granularity.total_seconds()) // 60) % 60),
                second=int((end_slot_index * granularity.total_seconds()) % 60),
            ),
            end=time(
                hour=int(((end_slot_index*granularity.total_seconds()+granularity.total_seconds()-1)//3600)%24),
                minute=int(((end_slot_index*granularity.total_seconds()+granularity.total_seconds()-1)//60)%60),
                second=int((end_slot_index*granularity.total_seconds()+granularity.total_seconds()-1)%60),
            ),
        )
        calendar[event.end.weekday()].add(end_slot)

    return calendar


def discover_calendar_by_resource(
        log: Log,
        granularity: timedelta=timedelta(minutes=15),
) -> typing.Mapping[Resource, WeeklyCalendar]:
    """
    Discover the availability calendars for each resource in the log.

    Parameters
    ----------
    * `log`:            *an event log*
    * `granularity`:    *the granularity used to compute the resources utilization rate

    Returns
    -------
    * a mapping with pairs (`resource`, `calendar`) where calendar is an iterable of booleans indicating if the resource
    was available at the interval i for intervals of size granularity
    """
    # build a map with the events executed by each resource
    events_per_resource = defaultdict(list)
    for event in log:
        events_per_resource[event.resource].append(event)

    # compute the calendar for each resource
    calendars = {}
    for (resource, events) in events_per_resource.items():
        calendars[resource] = __compute_calendar(events, granularity)

    return calendars


def apply_calendar_to_timeframe(
        timeframe: Interval,
        weekly_calendar: WeeklyCalendar,
) -> IntervalTree:
    """
    Apply a weekly calendar to a specific timeframe, generating availability intervals for the given interval

    Parameters
    ----------
    * `timeframe`: the timeframe where the availability calendar will be applied
    * `weekly_calendar`: the availability weekly calendar

    Returns
    -------
    * an interval tree with the availability periods within the given timeframe
    """
    # create an empty tree for storing the applied calendar
    tree = IntervalTree()
    # count the number of processed days
    processed_days = 0

    # for each day in the timeframe
    while (timeframe.begin + timedelta(days=processed_days)) <= timeframe.end:
        # the current date is the initial date + the already processed days
        current_date = timeframe.begin + timedelta(days=processed_days)

        # for each available slot in the current weekday
        for slot in weekly_calendar[current_date.weekday()]:
            # apply the slot to the current date
            slot_in_date: Interval = Interval(
                begin=current_date.replace(
                    hour=slot.begin.hour,
                    minute=slot.begin.minute,
                    second=slot.begin.second,
                    microsecond=0,
                ),
                end=current_date.replace(
                    hour=slot.end.hour,
                    minute=slot.end.minute,
                    second=slot.end.second,
                    microsecond=999999,
                ),
            )

            # if the slot overlaps the timeframe, save it
            if slot_in_date.overlaps(timeframe):
                tree.add(slot_in_date)

        # increment the count of already processed days
        processed_days += 1

    # once all availability slots have been applied, collapse the tree joining the neighbouring intervals
    # here we consider neighbouring intervals those separated at most by 1 second
    tree.merge_neighbors(
        distance = timedelta(seconds=1),
        strict = False,
    )

    return tree
