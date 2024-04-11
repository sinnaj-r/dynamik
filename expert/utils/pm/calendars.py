from __future__ import annotations

import itertools
import typing
from collections import defaultdict
from datetime import datetime, time, timedelta
from statistics import mean, stdev

import scipy
from intervaltree import Interval, IntervalTree

from expert.process_model import Log, Resource


class Calendar:
    """TODO docs"""

    owner: set[str]
    __calendar: dict[tuple[int, int], int]

    def __init__(self: typing.Self, owner: set[str] = frozenset(), calendar: dict[tuple[int, int], int] | None = None) -> None:
        self.owner = owner
        self.__calendar = calendar if calendar is not None else {slot: 0 for slot in itertools.product(range(7), range(24))}

    def __getitem__(self: typing.Self, key: int | tuple[int, int]) -> dict[int, int] | int:
        # if the key is an int, return all the slots for that weekday
        if isinstance(key, int):
            return {
                slot: self[slot] for slot in self.slots if slot[0] == key
            }
        # otherwise return the specific slot
        return self.__calendar[key] if key in self.__calendar else -1

    def __add__(self: typing.Self, other: Calendar) -> Calendar:
        return Calendar(
            owner=set(self.owner).union(other.owner),
            calendar={
                slot: self[slot] + other[slot] for slot in self.__calendar
            },
        )

    def transform(self: typing.Self, transformer: typing.Callable[[int], int]) -> Calendar:
        """TODO docs"""
        for slot in self.__calendar:
            self.__calendar[slot] = transformer(self.__calendar[slot])

        return self

    def apply(self: typing.Self, timeframe: Interval) -> IntervalTree:
        """
        Apply a calendar to a specific timeframe, generating availability intervals for the given interval

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
        # store the initial date
        initial_date = timeframe.begin.date()
        # count the total number of days in the timeframe
        total_days = (timeframe.end.date() - timeframe.begin.date()).days

        # iterate over all the days in the timeframe (+1 to make it end-inclusive)
        for elapsed_day in range(total_days + 1):
            # iterate over all hours in a day
            for hour in range(24):
                # current date is the initial date plus the already processed days
                current_date: datetime = (initial_date + timedelta(days=elapsed_day))
                # if a slot is present for the current weekday and hour, add a new interval to the applied calendar
                if self[(current_date.weekday(), hour)] > 0:
                    # create the applied slot, with the current date and the hour being processed
                    applied_slot = Interval(
                        begin=datetime.combine(
                            current_date,
                            time(hour=hour),
                            tzinfo=timeframe.begin.tzinfo,
                        ),
                        end=datetime.combine(
                            current_date,
                            time(hour=hour, minute=59, second=59, microsecond=999999),
                            tzinfo=timeframe.end.tzinfo,
                        ),
                    )
                    # if the applied slot overlaps the timeframe, add it to the calendar
                    # (all slots for a given day are generated, so some slots can not overlap the timeframe due to
                    # difference in time)
                    if applied_slot.overlaps(timeframe):
                        tree.add(applied_slot)

        # once all availability slots have been applied, simplify the tree joining the neighbouring intervals
        tree.merge_neighbors(distance=timedelta(seconds=1), strict=False)

        return tree

    def statistically_equals(self: typing.Self, other: Calendar, *, significance: float = 0.05) -> bool:
        """TODO docs"""
        results = {}
        for slot in self.slots:
            results[slot] = scipy.stats.poisson_means_test(
                self[slot],
                len(set(self.owner).union(other.owner)),
                other[slot],
                len(set(self.owner).union(other.owner)),
            )

        return all(result.pvalue >= significance for result in results.values())

    @property
    def slots(self: typing.Self) -> set[tuple[int, int]]:
        """TODO docs"""
        return set(self.__calendar.keys())

    @staticmethod
    def discover(log: Log) -> Calendar:
        """TODO docs"""
        # the calendar owners are the resources present in the log
        owner = {event.resource for event in log}

        # generate the initial calendar, where the keys are tuples in the form (week day, hour)
        calendar = {slot: 0 for slot in itertools.product(range(7), range(24))}

        # check intervals for each event in the log
        # we consider a resource is available in a given slot if any activity is recorded in that slot (start or end of activity instance)
        for event in log:
            calendar[(event.start.weekday(), event.start.time().hour)] += 1
            calendar[(event.end.weekday(), event.end.time().hour)] += 1

        # clean calendar by removing slots where value not in [avg-3sd, avg+3sd]
        average_observations_per_slot = mean(calendar.values())
        standard_deviation = stdev(calendar.values())

        # set as calendar the clean calendar
        return Calendar(
            owner=owner,
            calendar={
                slot: value if (average_observations_per_slot - 3 * standard_deviation) < value < (
                        average_observations_per_slot + 3 * standard_deviation) else 0
                for (slot, value) in calendar.items()
            }
        )


def discover_calendars(
        log: Log,
) -> typing.Mapping[Resource, Calendar]:
    """
    Discover the availability calendars for each resource in the log.

    Parameters
    ----------
    * `log`:            *an event log*

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
        calendars[resource] = Calendar.discover(events)

    return calendars
