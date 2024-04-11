"""This module provides some case-related utilities"""
import typing
from collections import defaultdict
from datetime import datetime

from expert.process_model import Event, Log


def compute_weekly_event_distribution(
        log: Log,
        *,
        filter_: typing.Callable[[Event], bool],
        extractor: typing.Callable[[Event], datetime],
) -> typing.Mapping[tuple[int, int], int]:
    """TODO docs"""
    events_per_hour_of_weekday: typing.Mapping[tuple[int, int], typing.Mapping[int, list]] = {
        (day, hour): defaultdict(list) for day in range(7) for hour in range(24)
    }

    for event in [event for event in log if filter_(event)]:
        events_per_hour_of_weekday[extractor(event).weekday(), extractor(event).hour][
            extractor(event).date().isocalendar().week].append(event)

    return {
        hour_of_weekday: sum(len(events) for events in events_per_week_of_year.values())
        for hour_of_weekday, events_per_week_of_year in events_per_hour_of_weekday.items()
    }

