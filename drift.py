from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd
from river.drift import ADWIN
from river.stream import iter_pandas

from fields import DEFAULT_EVENT_FIELDS
from model import Model, Event


def detect_drift(
        log: pd.DataFrame,
        timeframe_size: timedelta,
        initial_activity: str = 'START',
        final_activity: str = 'END'
) -> list[Event]:
    """
        Finds drifts in the cycle time of a process

        :param log: the input event log containing start and finish timestamps for each event
        :param timeframe_size: the size of the timeframe for the reference and drifting models
        :param initial_activity: the first activity of the process/process fragment to be monitored
        :param final_activity: the last activity of the process/process fragment to be monitored
        :return: a list of events causing drift
    """
    drifts = []
    drift_detector = ADWIN(delta=0.005)

    # Create the model with the given parameters
    model = Model(timeframe_size=timeframe_size, initial_activity=initial_activity, final_activity=final_activity)

    # Iterate over the events in the log
    for index, (source_event, _) in enumerate(iter_pandas(log)):
        # Create the Event instance from the content of the dictionary
        event = Event.from_dict(source=source_event, fields=DEFAULT_EVENT_FIELDS)
        logging.debug('processing event %(index)d: %(event)r', {'index': index, 'event': event})
        # Update the model with the new event
        case_duration_in_seconds = model.update(event=event)

        # If the model update returned a time measurement, add it to the drift detector instance
        if case_duration_in_seconds is not None:
            drift_detector.update(case_duration_in_seconds)

        # If a change is detected, save the event and reset the model and the drift detector
        if drift_detector.drift_detected:
            drifts.append(event)
            logging.info('drift detected at event %(index)d: %(event)r', {'index': index, 'event': event})
            # Reset models
            model.reset()

    return drifts
