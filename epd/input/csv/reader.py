import logging
from collections.abc import Iterator

import pandas as pd
from river.stream import iter_pandas

from epd.input import Mapping
from epd.input.csv import DEFAULT_APROMORE_CSV_MAPPING
from epd.model import Event


def read_csv_log(log_path: str, *, attribute_mapping: Mapping = DEFAULT_APROMORE_CSV_MAPPING,
                 filter_events_without_resources: bool = True) -> Iterator[Event]:
    """
    Read an event log from a CSV file.

    Reads an event log from a CSV file given the column IDs in [log_ids]. Set the start_time and
    end_time columns to date, filter events without a resource assigned and sort by [end, start].

    :param log_path: path to the CSV log file.
    :param attribute_mapping: an object defining a mapping between CSV columns and event attributes.
    :param filter_events_without_resources: a flag indicating if events without a resource should be removed

    :return: an iterator with the events from the read file
    """
    # Read log
    event_log = pd.read_csv(log_path, skipinitialspace=True)

    # Force case identifier to be a string
    event_log[attribute_mapping.case] = event_log[attribute_mapping.case].astype(str)

    # Convert timestamp value to pd.Timestamp, setting timezone to UTC
    event_log[attribute_mapping.start] = pd.to_datetime(event_log[attribute_mapping.start], utc=True)
    event_log[attribute_mapping.end] = pd.to_datetime(event_log[attribute_mapping.end], utc=True)

    # Filter out the events without a resource
    if filter_events_without_resources:
        event_log = event_log.dropna(subset=[attribute_mapping.resource])
    else:
        event_log[attribute_mapping.resource] = event_log[attribute_mapping.resource].fillna('UNKNOWN')

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
