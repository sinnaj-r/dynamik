"""
This module contains everything needed for reading an event log from a JSON file and produce a
`collections.abc.Generator[expert.model.Event, None, None]` that yields events one by one in order
to simulate an event stream.
"""

from .mapping import DEFAULT_JSON_MAPPING
from .reader import read_json_log

__all__ = ["read_json_log", "DEFAULT_JSON_MAPPING"]
__docformat__ = "markdown"
