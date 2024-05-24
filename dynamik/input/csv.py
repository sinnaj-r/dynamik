"""
This module contains everything needed for reading an event log from a CSV file.

The log is read as a `typing.Generator[dynamik.model.Event, None, None]` that yields events one by one in order to
simulate an event stream where events can be consumed only once.
"""
import typing
from datetime import timedelta

import numpy as np
import pandas as pd

from dynamik.input import EventMapping
from dynamik.model import Event, Log
from dynamik.utils.logger import LOGGER

DEFAULT_CSV_MAPPING: EventMapping = EventMapping(
    start="start",
    end="end",
    enablement="enabled",
    case="case",
    activity="activity",
    resource="resource",
)


def __preprocess_and_sort(
        event_log: pd.DataFrame,
        attribute_mapping: EventMapping,
        *,
        add_artificial_start_end_events: bool = False,
) -> typing.Generator[Event, None, None]:
    # Convert timestamp value to pd.Timestamp, setting timezone to UTC
    event_log[attribute_mapping.start.lower()] = pd.to_datetime(
        event_log[attribute_mapping.start.lower()],
        utc=True,
        format="ISO8601",
    ).dt.to_pydatetime()
    event_log[attribute_mapping.end.lower()] = pd.to_datetime(
        event_log[attribute_mapping.end.lower()],
        utc=True,
        format="ISO8601",
    ).dt.to_pydatetime()

    # add synthetic events to the start and end of traces if asked
    if add_artificial_start_end_events:
        start_mapping = {
            attribute_mapping.activity.lower(): (attribute_mapping.activity.lower(), lambda _: "__SYNTHETIC_START_EVENT__"),
            attribute_mapping.start.lower(): (attribute_mapping.start.lower(), lambda values: values.min() - timedelta(seconds=1)),
            attribute_mapping.end.lower(): (attribute_mapping.start.lower(), lambda values: values.min() - timedelta(seconds=1)),
        }

        end_mapping = {
            attribute_mapping.activity.lower(): (attribute_mapping.activity.lower(), lambda _: "__SYNTHETIC_END_EVENT__"),
            attribute_mapping.start.lower(): (attribute_mapping.end.lower(), lambda values: values.max() + timedelta(seconds=1)),
            attribute_mapping.end.lower(): (attribute_mapping.end.lower(), lambda values: values.max() + timedelta(seconds=1)),
        }

        if attribute_mapping.enablement is not None:
            start_mapping[attribute_mapping.enablement.lower()] = (attribute_mapping.start.lower(), lambda val: val.min() - timedelta(seconds=1))
            end_mapping[attribute_mapping.enablement.lower()] = (attribute_mapping.end.lower(), lambda val: val.max() + timedelta(seconds=1))

        start_events = event_log.groupby(attribute_mapping.case.lower(), as_index=False).agg(
            **start_mapping,
        )

        end_events = event_log.groupby(attribute_mapping.case.lower(), as_index=False).agg(
            **end_mapping,
        )

        event_log = pd.concat([event_log, start_events, end_events], ignore_index=True)

    # cast enablement times to datetime type if present
    if attribute_mapping.enablement is not None:
        event_log[attribute_mapping.enablement.lower()] = pd.to_datetime(
            event_log[attribute_mapping.enablement.lower()],
            utc=True,
            format="ISO8601",
        ).dt.to_pydatetime()

    # Sort events
    if attribute_mapping.enablement is not None:
        event_log = event_log.sort_values([attribute_mapping.end.lower(), attribute_mapping.start.lower(), attribute_mapping.enablement.lower()])
    else:
        event_log = event_log.sort_values([attribute_mapping.end.lower(), attribute_mapping.start.lower()])

    # Replace missing values with None
    event_log = event_log.replace(np.nan, None)

    # Print some debugging information about the parsed log
    LOGGER.info("    %d events", event_log.count()[attribute_mapping.case.lower()])
    LOGGER.info("    %d cases", event_log[attribute_mapping.case.lower()].unique().size)
    LOGGER.info("    %d activities", event_log[attribute_mapping.activity.lower()].unique().size)
    for activity in event_log[attribute_mapping.activity.lower()].unique():
        LOGGER.info('         %d "%s" instances',
                    len(event_log[event_log[attribute_mapping.activity.lower()] == activity]),
                    activity,
                    )
    LOGGER.info("    %d resources", event_log[attribute_mapping.resource.lower()].unique().size)
    LOGGER.info(
        "    timeframe from %s to %s (%s)",
        event_log[attribute_mapping.start.lower()].min(),
        event_log[attribute_mapping.end.lower()].max(),
        event_log[attribute_mapping.end.lower()].max() - event_log[attribute_mapping.start.lower()].min(),
        )

    malformed_events = len(
        event_log[event_log[attribute_mapping.start.lower()] > event_log[attribute_mapping.end.lower()]])

    if malformed_events > 0:
        LOGGER.error("    %d malformed events have been detected! Results may be inaccurate", malformed_events)

    # Yield parsed events
    yield from (attribute_mapping.tuple_to_event(row) for row in event_log.itertuples())


def read_csv_log(
        log_path: str,
        *,
        add_artificial_start_end_events: bool = False,
        attribute_mapping: EventMapping = DEFAULT_CSV_MAPPING,
        case_prefix: str = "",
        preprocessor: typing.Callable[[Log], Log] = lambda log: log,
) -> typing.Generator[Event, None, None]:
    """
    Read an event log from a CSV file.

    The file is expected to contain a header row and an event per row.
    Events will be mapped to `dynamik.model.Event` instances by applying the provided `dynamik.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the path to the CSV log file*
    * `attribute_mapping`:  *an instance of `dynamik.input.Mapping` defining a mapping between CSV columns and event attributes*.
    * `case_prefix`:        *a prefix that will be prepended to every case ID on the log*

    Yields
    ------
    * the parsed events sorted by the `dynamik.model.Event.end`and `dynamik.model.Event.start` timestamps and transformed to
      instances of `dynamik.model.Event`
    """
    # Read log
    event_log = pd.read_csv(log_path, skipinitialspace=True, na_values=["[NULL]", ""], engine="c")

    # Force column names to be lowercase
    event_log.columns = event_log.columns.str.lower()

    # Force case identifier to be a string and add prefix
    event_log[attribute_mapping.case.lower()] = str(f"{case_prefix}/") + event_log[
        attribute_mapping.case.lower()].astype(str)

    LOGGER.info("parsed logs from %s", log_path)

    event_log = __preprocess_and_sort(event_log, attribute_mapping, add_artificial_start_end_events=add_artificial_start_end_events)

    if preprocessor is not None:
        event_log = preprocessor(event_log)

    yield from event_log


def read_and_merge_csv_logs(
        logs: typing.Iterable[str],
        *,
        add_artificial_start_end_events: bool = False,
        attribute_mapping: EventMapping = DEFAULT_CSV_MAPPING,
        preprocessor: typing.Callable[[Log], Log] = lambda log: log,
) -> typing.Generator[Event, None, None]:
    """
    Read a set of event logs from CSV files and combine them.

    The files are expected to contain a header row and an event per row.
    Events will be mapped to `dynamik.model.Event` instances by applying the provided `dynamik.input.Mapping` object.
    The functon returns a Generator that yields the events from the log file one by one to optimize memory usage.

    Parameters
    ----------
    * `log_path`:           *the collection of pairs (log name, path to the CSV log file). If the logs have the same
                             name the events with the same case id will be considered as part of the same case.
                             Otherwise, the log name will be appended to the case id so events from different log files
                             are considered as different cases even if they have the same case id*
    * `attribute_mapping`:  *an instance of `dynamik.input.Mapping` defining a mapping between CSV columns and event
                             attributes*

    Yields
    ------
    * the parsed events sorted by the `dynamik.model.Event.end`and `dynamik.model.Event.start` timestamps and transformed
      to instances of `dynamik.model.Event`
    """
    event_logs = []

    # Read logs
    for file in logs:
        log = pd.read_csv(file, skipinitialspace=True, na_values=["[NULL]", ""], engine="c")
        # Force column names to be lowercase
        log.columns = log.columns.str.lower()
        # Force case identifier to be a string
        log[attribute_mapping.case.lower()] = log[attribute_mapping.case.lower()].astype(str)
        event_logs.append(log)

    LOGGER.info("parsed logs from %s:", logs)

    concatenated_logs = pd.concat(event_logs, ignore_index=True)

    event_log = __preprocess_and_sort(
        concatenated_logs,
        attribute_mapping,
        add_artificial_start_end_events=add_artificial_start_end_events,
    )

    if preprocessor is not None:
        event_log = preprocessor(event_log)

    yield from event_log
