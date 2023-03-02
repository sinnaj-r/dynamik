"""
JSON input module.

This module contains everything needed for reading an event log from a JSON file and produce a Generator that yields
events one by one in order to simulate an event stream.
"""

from .mapping import DEFAULT_JSON_MAPPING
from .reader import read_json_log

__all__ = ["DEFAULT_JSON_MAPPING", "read_json_log"]
