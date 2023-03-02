import logging
from collections.abc import Iterator

import pandas as pd
from river.stream import iter_pandas

from epd.input import Mapping
from epd.input.csv import DEFAULT_CSV_MAPPING
from epd.model import Event


def read_csv_log(log_path: str, *, attribute_mapping: Mapping = DEFAULT_CSV_MAPPING) -> Iterator[Event]:
    """
    Read an event log from a CSV file.

    The file is expected to contain a header row and an event per row.
    Events will be mapped to `epd.model.Event` instances by applying the provided `epd.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the path to the CSV log file*
    * `attribute_mapping`:  *an instance of `epd.input.Mapping` defining a mapping between CSV columns and event attributes*.

    Yields
    ------
    * the parsed events sorted by the `epd.model.Event.end`and `epd.model.Event.start` timestamps and transformed to
      instances of `epd.model.Event`

    Returns
    -------
    * a `collections.abc.Generator[epd.model.Event, None, None]` containing the events from the read file
    """
    # Read log
    event_log = pd.read_csv(log_path, skipinitialspace=True)

    # Force case identifier to be a string
    event_log[attribute_mapping.case] = event_log[attribute_mapping.case].astype(str)

    # Convert timestamp value to pd.Timestamp, setting timezone to UTC
    event_log[attribute_mapping.start] = pd.to_datetime(event_log[attribute_mapping.start], utc=True)
    event_log[attribute_mapping.end] = pd.to_datetime(event_log[attribute_mapping.end], utc=True)

    # Sort events
    event_log = event_log.sort_values([attribute_mapping.end, attribute_mapping.start])

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
    yield from (attribute_mapping.dict_to_event(event_dict) for event_dict, _ in iter_pandas(event_log))
