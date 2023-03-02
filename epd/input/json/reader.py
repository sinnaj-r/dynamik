import json
from collections.abc import Generator, Iterator

from epd.input import Mapping
from epd.input.json import DEFAULT_JSON_MAPPING
from epd.model import Event


def read_json_log(log_path: str, *, attribute_mapping: Mapping = DEFAULT_JSON_MAPPING) -> Iterator[Event]:
    """
    Read an event log from a JSON file.

    Reads an event log from a JSON file with an array of events, filter events without a resource assigned and sort
    by end.

    :param log_path: path to the CSV log file.
    :param attribute_mapping: an object defining a mapping between CSV columns and event attributes.

    :return: an iterator containing the events from the read file
    """
    # Read log
    with open(log_path) as file:
        content: Generator[Event, None, None] = (attribute_mapping.dict_to_event(source) for source in json.load(file))
        # Yield events from the parsed file
        yield from sorted(content, key=lambda event: event.end)
