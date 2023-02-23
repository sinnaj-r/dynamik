import logging

import pandas as pd

from fields import EventFields, DEFAULT_EVENT_FIELDS


def read_csv_log(
        log_path: str,
        event_fields: EventFields = DEFAULT_EVENT_FIELDS
) -> pd.DataFrame:
    """
        Read an event log from a CSV file given the column IDs in [log_ids]. Set the enabled_time, start_time, and
        end_time columns to date,set the NA resource cells to [missing_value] if not None, and sort by [end, start,
        enabled].

        :param log_path: path to the CSV log file.
        :param event_fields: the event field identifiers in the CSV log file.

        :return: the read event log
    """
    # Read log
    event_log = pd.read_csv(log_path, skipinitialspace=True)

    # Force case identifier to be a string
    event_log[event_fields.CASE] = event_log[event_fields.CASE].astype(str)

    # Convert timestamp value to pd.Timestamp (setting timezone to UTC)
    event_log[event_fields.START] = pd.to_datetime(event_log[event_fields.START], utc=True)
    event_log[event_fields.END] = pd.to_datetime(event_log[event_fields.END], utc=True)

    # Sort events
    event_log = event_log.sort_values([event_fields.END, event_fields.START])

    # Replace missing resources with 'NONE'
    event_log[event_fields.RESOURCE].fillna('NONE', inplace=True)

    # Print some debugging information about the parsed log
    logging.debug('parsed log from %(log_path)s:', {'log_path': log_path})
    logging.debug('\t- %(count)d events', {'count': event_log.count()[event_fields.CASE]})
    logging.debug('\t- %(count)d activities', {'count': event_log[event_fields.ACTIVITY].unique().size})
    logging.debug('\t- %(count)d resources', {'count': event_log[event_fields.RESOURCE].unique().size})
    logging.debug('\t- %(timeframe)s timeframe',
                  {'timeframe': event_log[event_fields.END].max() - event_log[event_fields.START].min()})
    logging.debug('\t\t (from %(start)s to %(end)s)',
                  {'start': event_log[event_fields.START].min(), 'end': event_log[event_fields.END].max()})

    # Return parsed event log
    return event_log
