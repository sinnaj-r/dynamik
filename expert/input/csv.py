"""
This module contains everything needed for reading an event log from a CSV file.

The log is read as a `typing.Generator[expert.model.Event, None, None]` that yields events one by one in order to
simulate an event stream where events can be consumed only once.
"""
import logging
import typing

import numpy as np
import pandas as pd
from river.stream import iter_pandas

from expert.input import EventMapping
from expert.model import Event

DEFAULT_CSV_MAPPING: EventMapping = EventMapping(start="start", end="end", case="case", activity="activity", resource="resource")

DEFAULT_APROMORE_CSV_MAPPING: EventMapping = EventMapping(start="start_time", end="end_time", case="case_id",
                                                          activity="Activity", resource="Resource")

def read_csv_log(log_path: str, *,
                 attribute_mapping: EventMapping = DEFAULT_CSV_MAPPING,
                 case_prefix: str = "") -> typing.Generator[Event, None, None]:
    """
    Read an event log from a CSV file.

    The file is expected to contain a header row and an event per row.
    Events will be mapped to `expert.model.Event` instances by applying the provided `expert.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the path to the CSV log file*
    * `attribute_mapping`:  *an instance of `expert.input.Mapping` defining a mapping between CSV columns and event
                             attributes*.
    * `case_prefix`:        *a prefix that will be prepended to every case ID on the log*

    Yields
    ------
    * the parsed events sorted by the `expert.model.Event.end`and `expert.model.Event.start` timestamps and transformed
      to instances of `expert.model.Event`
    """
    # Read log
    event_log = pd.read_csv(log_path, skipinitialspace=True)

    # Force case identifier to be a string and add prefix
    event_log[attribute_mapping.case] = str(case_prefix) + event_log[attribute_mapping.case].astype(str)

    # Convert timestamp value to pd.Timestamp, setting timezone to UTC
    event_log[attribute_mapping.start] = pd.to_datetime(event_log[attribute_mapping.start], utc=True)
    event_log[attribute_mapping.end] = pd.to_datetime(event_log[attribute_mapping.end], utc=True)

    # Sort events
    event_log = event_log.sort_values([attribute_mapping.end, attribute_mapping.start])

    # Replace missing values with None
    event_log = event_log.replace(np.nan, None)

    # Print some debugging information about the parsed log
    logging.debug('parsed log from %(log_path)s:', {'log_path': log_path})
    logging.debug('\t- %(count)d events', {'count': event_log.count()[attribute_mapping.case]})
    logging.debug('\t- %(count)d activities',
                  {'count': event_log[attribute_mapping.activity].unique().size})
    logging.debug('\t- %(count)d resources',
                  {'count': event_log[attribute_mapping.resource].unique().size})
    logging.debug('\t- %(timeframe)s timeframe',
                  {'timeframe': event_log[attribute_mapping.end].max() - event_log[
                      attribute_mapping.start].min()})
    logging.debug('\t\t (from %(start)s to %(end)s)',
                  {'start': event_log[attribute_mapping.start].min(),
                   'end': event_log[attribute_mapping.end].max()})

    # Yield parsed events
    yield from (attribute_mapping.dict_to_event(evt) for evt, _ in iter_pandas(event_log))
