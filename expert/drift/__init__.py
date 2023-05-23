"""This module contains the functions needed for detecting and explaining performance drifts in a process model."""
from __future__ import annotations

import typing
from datetime import timedelta

import scipy

from expert.drift.model import NO_DRIFT, Drift, DriftLevel, DriftModel
from expert.logger import LOGGER
from expert.model import Activity, Log, Test


def default_drift_detector_test_factory(alpha: float = 0.05) -> Test:
    """Default statistical test factory used for comparing the reference and running distributions using the U test"""
    def __test_drift(reference: typing.Iterable[float], running: typing.Iterable[float]) -> bool:
        p_value = scipy.stats.mannwhitneyu(list(reference), list(running), alternative="less").pvalue

        LOGGER.verbose("test(ct reference < ct running) p-value: %.4f", p_value)

        return p_value < alpha
    return __test_drift


def detect_drift(
        log: Log,
        *,
        initial_activities: typing.Iterable[Activity] = tuple("START"),
        final_activities: typing.Iterable[Activity] = tuple("END"),
        timeframe_size: timedelta,
        warm_up: timedelta,
        overlap_between_models: timedelta = timedelta(),
        test: Test = default_drift_detector_test_factory(),
        warnings_to_confirm: int = 5,
) -> typing.Generator[Drift, None, typing.Iterable[Drift]]:
    """Find and explain drifts in the performance of a process execution by monitoring its cycle time.

    The detection follows these steps:

    1. A reference model is built using the first events from the log. This model will be used later as the ground
       truth to check for the change causes.
    2. A running model is defined, using the last events from the log. This is the model that will be checked for
       changes.
    3. Events are read one by one, and the running model is updated, computing the cycle time for each completed
       case. This cycle time are monitored using a statistical test.
    4. If a change is detected, the reference and running models are reset and the detection process starts again.

    Parameters
    ----------
    * `log`:                    *the input event log*
    * `timeframe_size`:         *the size of the timeframe for the reference and running models*
    * `warm_up`:                *the size of the warm-up where events will be discarded*
    * `overlap_between_models`: *the overlapping between running models (must be smaller than the timeframe size).
                                 Negative values imply leaving a space between successive models.*
    * `test`:                   *the test for evaluating if there are any difference between the reference and the
                                 running models*
    * `initial_activity`:       *the first activity of the process/process fragment to be monitored*
    * `final_activity`:         *the last activity of the process/process fragment to be monitored*
    * `warnings_to_confirm`:    *the number of consecutive drift warnings to confirm a change*

    Yields
    ------
    * each confirmed drift model

    Returns
    -------
    * the list of detected and confirmed drifts
    """
    LOGGER.info("detecting drift with params:")
    LOGGER.info("    timeframe size: %s", timeframe_size)
    LOGGER.info("    overlapping: %s", overlap_between_models)
    LOGGER.info("    warm up: %s", warm_up)
    LOGGER.info("    warnings before confirmation: %s", warnings_to_confirm)

    # Create a list for storing the drifts
    drifts: list[Drift] = []

    # Create the model with the given parameters
    drift_model = DriftModel(
        timeframe_size=timeframe_size,
        initial_activities=initial_activities,
        final_activities=final_activities,
        warm_up=warm_up,
        test=test,
        warnings_to_confirm = warnings_to_confirm,
        overlap_between_models=overlap_between_models,
    )

    # Store the event causing the first drift warning for localization
    first_warning: tuple | None = None

    # Iterate over the events in the log
    for index, event in enumerate(log):
        # Discard the event if it is not valid
        if not event.is_valid():
            LOGGER.warning("malformed event %r will be discarded", event)
            LOGGER.warning("    event validity violations: %r", event.violations)
            continue

        # Update the model with the new event
        drift_model.update(event)

        # Check the drift status
        if drift_model.drift == NO_DRIFT:
            # If there are no drifts, reset the warnings
            first_warning = None
        elif first_warning is None:
            # If the first warning, store the position
            first_warning = (index, event, drift_model.drift)

        if drift_model.drift.level == DriftLevel.CONFIRMED:
            # If the drift is confirmed, save the drift and reset the model
            drifts.append(drift_model.drift)
            LOGGER.notice("drift detected at event %d: %r", first_warning[0], first_warning[1])
            LOGGER.info("    confirmed at event %d: %r)", index, event)
            # Yield the drift
            yield drift_model.drift
            # Reset the first warning (the change has already been confirmed)
            first_warning = None
            # Reset the model
            drift_model.reset()

    return drifts
