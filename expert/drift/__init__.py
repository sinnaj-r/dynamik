"""This module contains the functions needed for detecting and explaining performance drifts in a process model."""
from __future__ import annotations

import logging
import typing
from datetime import timedelta

from expert.drift.model import Model
from expert.model import Event
from expert.utils.statistical_tests import ks_test


def detect_drift(
        log: typing.Generator[Event, typing.Any, typing.Any],
        timeframe_size: timedelta,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool] = ks_test(),
        initial_activity: str = "START",
        final_activity: str = "END",
        filters: typing.Iterable[typing.Callable[[Event], bool]] = (),
        events_between_evaluations: int = 1,
) -> typing.Generator[Model, None, list[Model]]:
    """Find and explain drifts in the performance of a process execution by monitoring its cycle time.

    The detection follows these steps:
    1. A reference model is built using the first events from the log. This model will be used later as the ground
       truth to check for the change causes.
    2. A running model is defined, using the last events from the log. This is the model that will be checked for
       changes.
    3. Events are read one by one, and the running model is updated, computing the cycle time for each completed
       case. This cycle time are monitored using a statistical test.
    4. If a change in the process cycle time is detected by the statistical test, the causes leading to that change are
       analyzed

    Parameters
    ----------
    * `log`:                        *the input event log*
    * `timeframe_size`:             *the size of the timeframe for the reference and running models*
    * `test`:                       *the test for evaluating if there are any difference between the reference and the
                                     running models*
    * `initial_activity`:           *the first activity of the process/process fragment to be monitored*
    * `final_activity`:             *the last activity of the process/process fragment to be monitored*
    * `filters`:                    *a collection of filters that will be applied to the events before processing them*
    * `events_between_evaluations`  *the number of events between consecutive evaluations of the statistical test*

    Yields
    ------
    * each detected drift model

    Returns
    -------
    * the list of detected drifts
    """
    # Create a list for storing the drifts
    drifts = []

    # Create the model with the given parameters
    drift_model = Model(timeframe_size=timeframe_size, initial_activity=initial_activity,final_activity=final_activity)

    # Iterate over the events in the log
    for index, event in enumerate(log):
        if not event.enabled <= event.start <= event.end:
            logging.warning(
                "detected malformed event %(event)s (violation: %(violation)s). event will be discarded",
                {
                    "event": event,
                    "violation": "enabled > start" if event.enabled > event.start
                                                   else "start > end" if event.start > event.end
                                                                      else "enabled > end",
                },
            )
            continue
        # Discard the event if it does not satisfy any of the conditions defined in the filters
        if any(not event_filter(event) for event_filter in filters):
            continue

        # Update the model with the new event
        drift_model.update(event)

        # Evaluate if the reference model is ready and events_between_evaluations have passed since the last evaluation
        # If the test detects a drift, notify and return
        if drift_model.model_ready and index % events_between_evaluations == 0 and test(
            list(drift_model.reference_model_durations.values()), list(drift_model.running_model_durations.values()),
        ):
            logging.info("drift detected at event %(index)d: %(event)r", {"index": index, "event": event})
            # Save the drift
            drifts.append(drift_model)
            # Yield the drift with the causes for the change
            yield drift_model
            # Reset the drift model
            drift_model = Model(timeframe_size=timeframe_size, initial_activity=initial_activity,
                                final_activity=final_activity)

    return drifts
