import json
from collections.abc import Generator, Iterator

from expert.input import Mapping
from expert.model import Event
from .mapping import DEFAULT_JSON_MAPPING


def read_json_log(log_path: str, *, attribute_mapping: Mapping = DEFAULT_JSON_MAPPING) -> Iterator[Event]:
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

    Returns
    -------
    * a collections.abc.Generator[expert.model.Event, None, None]` containing the events from the read file
    """
    # Read log
    with open(log_path) as file:
        content: Generator[Event, None, None] = (attribute_mapping.dict_to_event(source) for source in json.load(file))
        # Yield events from the parsed file
        yield from sorted(content, key=lambda event: event.end)
