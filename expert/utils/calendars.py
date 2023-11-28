import collections
import typing
from collections import defaultdict
from datetime import date, datetime, time, timedelta

from intervaltree import Interval, IntervalTree

from expert.model import Log, Resource, WeeklyCalendar


def __compute_calendar(
        log: Log,
        granularity: timedelta = timedelta(hours=1),
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
            begin=datetime.combine(date.min, time(
                hour=int(((start_slot_index * granularity.total_seconds()) // 3600) % 24),
                minute=int(((start_slot_index * granularity.total_seconds()) // 60) % 60),
                second=int((start_slot_index * granularity.total_seconds()) % 60),
            )),
            # the slot ends at index * granularity + granularity
            # subtracting 1 second we ensure we get "correct" intervals and prevent the case where (23:45 > 00:00)
            end=datetime.combine(date.min, time(
                hour=int(
                    ((start_slot_index * granularity.total_seconds() + granularity.total_seconds() - 1) // 3600) % 24),
                minute=int(
                    ((start_slot_index * granularity.total_seconds() + granularity.total_seconds() - 1) // 60) % 60),
                second=int((start_slot_index * granularity.total_seconds() + granularity.total_seconds() - 1) % 60),
            )),
            data=1.0,
        )
        # add the slot to the resource calendar for the corresponding weekday
        calendar[event.start.weekday()].add(start_slot)

        # repeat the same procedure as before, but for the ending time of the event
        end = event.end.time()
        end_slot_index = int((end.second + end.minute * 60 + end.hour * 60 * 60) // granularity.total_seconds())
        end_slot = Interval(
            begin=datetime.combine(date.min, time(
                hour=int(((end_slot_index * granularity.total_seconds()) // 3600) % 24),
                minute=int(((end_slot_index * granularity.total_seconds()) // 60) % 60),
                second=int((end_slot_index * granularity.total_seconds()) % 60),
            )),
            end=datetime.combine(date.min, time(
                hour=int(
                    ((end_slot_index * granularity.total_seconds() + granularity.total_seconds() - 1) // 3600) % 24),
                minute=int(
                    ((end_slot_index * granularity.total_seconds() + granularity.total_seconds() - 1) // 60) % 60),
                second=int((end_slot_index * granularity.total_seconds() + granularity.total_seconds() - 1) % 60),
            )),
            data=1.0,
        )
        calendar[event.end.weekday()].add(end_slot)

    reduced_calendar: dict[int, collections.Set] = defaultdict(set)
    # transform the calendar sets to an intervaltree and join neighbouring intervals
    for (week_day, intervals) in calendar.items():
        tree = IntervalTree(intervals)
        tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)
        reduced_calendar[week_day] = tree

    return reduced_calendar


def discover_calendars(
        log: Log,
        granularity: timedelta = timedelta(hours=1),
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

    # count the total number of days in the timeframe
    total_days = (timeframe.end.date() - timeframe.begin.date()).days

    # store the initial date
    initial_date = timeframe.begin.date()

    # for each day in the timeframe
    while processed_days <= total_days:
        # the current date is the initial date + the already processed days
        current_date = initial_date + timedelta(days=processed_days)

        # for each available slot in the current weekday
        for slot in weekly_calendar[current_date.weekday()]:
            # apply the slot to the current date
            slot_in_date: Interval = Interval(
                begin=datetime.combine(
                    current_date,
                    time(hour=slot.begin.hour, minute=slot.begin.minute, second=slot.begin.second),
                    tzinfo=timeframe.begin.tzinfo,
                ),
                end=datetime.combine(
                    current_date,
                    time(hour=slot.end.hour, minute=slot.end.minute, second=slot.end.second),
                    tzinfo=timeframe.end.tzinfo,
                ),
            )

            # if the slot overlaps the timeframe, save it
            if slot_in_date.overlaps(timeframe):
                tree.add(slot_in_date)

        # increment the count of already processed days
        processed_days += 1

    # once all availability slots have been applied, collapse the tree joining the neighbouring intervals
    # here we consider neighbouring intervals those separated at most by 1 second
    tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)

    return tree


def compute_weekly_available_time_per_resource(
        log: Log,
        granularity: timedelta = timedelta(hours=1),
) -> typing.Mapping[Resource, timedelta]:
    """TODO DOCS"""
    calendars_by_resource = discover_calendars(log, granularity)
    weekly_availability = defaultdict(timedelta)

    for resource in calendars_by_resource:
        for day in calendars_by_resource[resource]:
            weekly_availability[resource] += sum(
                (interval.length() for interval in calendars_by_resource[resource][day]),
                start=timedelta(),
            )

    return weekly_availability
