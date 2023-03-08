"""
This module contains everything needed for reading an event log from a JSON file.

The log is read and objects transformed `expert.model.Event` instances that are yield by a
`typing.Generator[expert.model.Event, None, None]` that simulates an event stream where events can be consumed only
once.
"""
import json
import typing

from expert.input import EventMapping
from expert.model import Event

DEFAULT_JSON_MAPPING = EventMapping(start="start", end="end", case="case", activity="activity", resource="resource")

def read_json_log(log_path: str, *,
                  attribute_mapping: EventMapping = DEFAULT_JSON_MAPPING) -> typing.Generator[Event, None, None]:
    """
    Read an event log from a JSON file.

    The file is expected to contain an array of events that will be mapped to `expert.model.Event` instances by applying
    the provided `expert.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the path to the JSON log file*
    * `attribute_mapping`:  *an instance of `Mapping` defining a mapping between JSON fields and event attributes*.

    Yields
    ------
    * the parsed events sorted by the `expert.model.Event.end` timestamp and transformed to instances of `expert.model.Event`
    """
    # Read log
    with open(log_path) as file:
        event_log = [attribute_mapping.dict_to_event(source) for source in json.load(file)]
        event_log = sorted(event_log, key=lambda event: event.end)

        # Yield events from the parsed file
        yield from event_log
