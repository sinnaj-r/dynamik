"""
This module contains everything needed for reading an event log from a CSV file.

The log is read as a `typing.Generator[expert.model.Event, None, None]` that yields events one by one in order to
simulate an event stream where events can be consumed only once.
"""
import typing

import numpy as np
import pandas as pd

from expert.input import EventMapping
from expert.logger import LOGGER
from expert.model import Event

DEFAULT_CSV_MAPPING: EventMapping = EventMapping(start="start", end="end", case="case", activity="activity", resource="resource")

DEFAULT_APROMORE_CSV_MAPPING: EventMapping = EventMapping(start="start_time", end="end_time", case="case_id",
                                                          activity="Activity", resource="Resource")


def __preprocess_and_yield(event_log: pd.DataFrame,
                           attribute_mapping: EventMapping) -> typing.Generator[Event, None, None]:
    # Convert timestamp value to pd.Timestamp, setting timezone to UTC
    event_log[attribute_mapping.start] = pd.to_datetime(event_log[attribute_mapping.start], utc=True)
    event_log[attribute_mapping.end] = pd.to_datetime(event_log[attribute_mapping.end], utc=True)

    # Sort events
    event_log = event_log.sort_values([attribute_mapping.end, attribute_mapping.start])

    # Replace missing values with None
    event_log = event_log.replace(np.nan, None)

    # Print some debugging information about the parsed log
    LOGGER.info("    - %d events", event_log.count()[attribute_mapping.case])
    LOGGER.info("    - %d cases", event_log[attribute_mapping.case].unique().size)
    LOGGER.info("    - %d activities", event_log[attribute_mapping.activity].unique().size)
    LOGGER.info("    - %d resources", event_log[attribute_mapping.resource].unique().size)
    LOGGER.info(
        "    - from %s to %s (%s)",
        event_log[attribute_mapping.start].min(),
        event_log[attribute_mapping.end].max(),
        event_log[attribute_mapping.end].max() - event_log[attribute_mapping.start].min(),
    )

    # Yield parsed events
    yield from (attribute_mapping.dict_to_event(evt._asdict()) for evt in event_log.itertuples())


def read_csv_log(
        log_path: str,
        *,
        attribute_mapping: EventMapping = DEFAULT_CSV_MAPPING,
        case_prefix: str = "",
        preprocessor: typing.Callable[[typing.Iterable[Event]], typing.Iterable[Event]] = lambda log: log,
) -> typing.Generator[Event, None, None]:
    """
    Read an event log from a CSV file.

    The file is expected to contain a header row and an event per row.
    Events will be mapped to `expert.model.Event` instances by applying the provided `expert.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the path to the CSV log file*
    * `attribute_mapping`:  *an instance of `expert.input.Mapping` defining a mapping between CSV columns and event attributes*.
    * `case_prefix`:        *a prefix that will be prepended to every case ID on the log*

    Yields
    ------
    * the parsed events sorted by the `expert.model.Event.end`and `expert.model.Event.start` timestamps and transformed to
      instances of `expert.model.Event`
    """
    # Read log
    event_log = pd.read_csv(log_path, skipinitialspace=True)

    # Force case identifier to be a string and add prefix
    event_log[attribute_mapping.case] = str(f"{case_prefix}/") + event_log[attribute_mapping.case].astype(str)

    LOGGER.info("parsed logs from %s", log_path)

    if preprocessor is not None:
        event_log = preprocessor(__preprocess_and_yield(event_log, attribute_mapping))
    else:
        event_log = __preprocess_and_yield(event_log, attribute_mapping)

    yield from event_log



def read_and_merge_csv_logs(
        logs: typing.Iterable[tuple],
        *,
        attribute_mapping: EventMapping = DEFAULT_CSV_MAPPING,
        preprocessor: typing.Callable[[typing.Iterable[Event]], typing.Iterable[Event]] = lambda log: log,
) -> typing.Generator[Event, None, None]:
    """
    Read a set of event logs from CSV files and combine them.

    The files are expected to contain a header row and an event per row.
    Events will be mapped to `expert.model.Event` instances by applying the provided `expert.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the collection of pairs (log name, path to the CSV log file). If the logs have the same
                             name the events with the same case id will be considered as part of the same case.
                             Otherwise, the log name will be appended to the case id so events from different log files
                             are considered as different cases even if they have the same case id*
    * `attribute_mapping`:  *an instance of `expert.input.Mapping` defining a mapping between CSV columns and event
                             attributes*

    Yields
    ------
    * the parsed events sorted by the `expert.model.Event.end`and `expert.model.Event.start` timestamps and transformed
      to instances of `expert.model.Event`
    """
    event_logs = []

    # Read logs
    for name, file in logs:
        log = pd.read_csv(file, skipinitialspace=True)
        # Force case identifier to be a string and add prefix
        log[attribute_mapping.case] = f"{name}/" + log[attribute_mapping.case].astype(str)
        event_logs.append(log)

    # Concat them into a single dataframe
    event_log = pd.concat(event_logs)

    LOGGER.info("parsed logs from %s:", logs)

    if preprocessor is not None:
        event_log = preprocessor(__preprocess_and_yield(event_log, attribute_mapping))
    else:
        event_log = __preprocess_and_yield(event_log, attribute_mapping)

    yield from event_log

