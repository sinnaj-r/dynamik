"""This module contains the functions needed for detecting and explaining performance drifts in a process model."""
from __future__ import annotations

import typing
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta

import scipy

from expert.drift.model import NO_DRIFT, Drift, DriftLevel, Model
from expert.model import Event, Log
from expert.utils.logger import LOGGER


def detect_drift(
        log: Log,
        *,
        timeframe_size: timedelta,
        warm_up: timedelta,
        overlap_between_models: timedelta = timedelta(),
        warnings_to_confirm: int = 5,
) -> typing.Generator[Drift, None, typing.Iterable[Drift]]:
    """Find drifts in the performance of a process execution by monitoring its cycle time.

    The detection follows these steps:

    1. A reference model is built using the first events from the log. This model will be used later as the ground
       truth to check for the change causes.
    2. A running model is defined, using the last events from the log. This is the model that will be checked for
       changes.
    3. Events are read one by one, and the running model is updated, computing the cycle time for each activity
    instance. These cycle times are monitored using a statistical test.
    4. If a change is detected, reset the reference and running models and the detection process starts again.

    Parameters
    ----------
    * `log`:                    *the input event log*
    * `window_size`:            *the size of the timeframe for the reference and running models*
    * `warm_up`:                *the size of the warm-up where events will be discarded*
    * `overlap_between_models`: *the overlapping between running models (must be smaller than the timeframe size).
                                 Negative values imply leaving a space between successive models.*
    * `warnings_to_confirm`:    *the number of consecutive drift warnings to confirm a change*

    Yields
    ------
    * each confirmed drift model

    Returns
    -------
    * the list of detected and confirmed drifts
    """
    LOGGER.notice("detecting drift with params:")
    LOGGER.notice("    timeframe size: %s", timeframe_size)
    LOGGER.notice("    overlapping: %s", overlap_between_models)
    LOGGER.notice("    warm up: %s", warm_up)
    LOGGER.notice("    warnings before confirmation: %s", warnings_to_confirm)

    # Create a list for storing the drifts
    drifts: list[Drift] = []

    # Create the model with the given parameters
    drift_detector = DriftDetector(
        timeframe_size=timeframe_size,
        warm_up=warm_up,
        warnings_to_confirm=warnings_to_confirm,
        overlap_between_models=overlap_between_models,
    )

    # Iterate over the events in the log
    for event in log:
        # Discard the event if it is not valid
        if not event.is_valid():
            LOGGER.warning("malformed event %r will be discarded", event)
            LOGGER.warning("    event validity violations: %r", event.violations)
            continue

        # Update the model with the new event
        drift = drift_detector.update(event)

        if drift.level == DriftLevel.CONFIRMED:
            # If the drift is confirmed, save the drift and reset the model
            drifts.append(drift)
            LOGGER.notice(
                "drift detected between %r and %r",
                drift.reference_model, drift.running_model,
            )
            LOGGER.info(
                "first drift warning between %r and %r",
                drift.first_warning.reference_model, drift.first_warning.running_model,
            )
            # Yield the drift
            yield drift

    return drifts


class DriftDetector:
    """Stores the model that will be used to detect drifts in the process."""

    # The size of the reference and running models, in time units
    __timeframe_size: timedelta
    # The number of drift warnings to confirm a drift
    __warnings_to_confirm: int = 0
    # The period considered as a warm-up
    __warm_up: timedelta
    # The overlap between running models
    __overlap: timedelta = timedelta()
    # The reference model
    __reference_model: Model | None = None
    # The running model
    __running_model: Model | None = None
    # The collection of detection results
    __drift_warnings: typing.MutableSequence[Drift] = deque([NO_DRIFT], maxlen=1)

    def __init__(
            self: typing.Self,
            *,
            timeframe_size: timedelta,
            warm_up: timedelta = timedelta(),
            overlap_between_models: timedelta = timedelta(),
            warnings_to_confirm: int = 3,
    ) -> None:
        """
        Create a new empty drift detection model with the given timeframe size and limit activities.

        Parameters
        ----------
        * `timeframe_size`:         *the timeframe used to build the reference and running models*
        * `warm_up`:                *the warm-up period during which events will be discarded*
        * `overlap_between_models`: *the overlapping between running models (must be smaller than the timeframe size)*
        * `warnings_to_confirm`:    *the number of consecutive detections needed for confirming a drift*
        """
        self.__timeframe_size = timeframe_size
        self.__warm_up = warm_up
        self.__warnings_to_confirm = warnings_to_confirm
        self.__drift_warnings = deque([NO_DRIFT] * warnings_to_confirm, maxlen=warnings_to_confirm)
        self.__overlap = overlap_between_models

    def __initialize_models(self: typing.Self, start: datetime) -> None:
        self.__reference_model = Model(start + self.__warm_up, self.__timeframe_size)
        self.__running_model = Model(start + self.__warm_up + self.__timeframe_size - self.__overlap, self.__timeframe_size)

        LOGGER.debug(
            "initializing models to timeframes (%s - %s) and (%s - %s)",
            self.__reference_model.start, self.__reference_model.end, self.__running_model.start, self.__running_model.end,
        )

    def __update_reference_model(self: typing.Self, event: Event) -> None:
        LOGGER.debug("updating reference model with event %r", event)
        # Append the event to the list of events in the reference model
        self.__reference_model.add(event)

    def __update_running_model(self: typing.Self, event: Event) -> None:
        LOGGER.debug("updating running model with event %r", event)
        # Append the event to the list of events in the running model
        self.__running_model.add(event)

    def __update_drifts(self: typing.Self) -> Drift:
        LOGGER.debug("updating drifts")

        LOGGER.verbose(
            "reference time distribution is %s",
            scipy.stats.describe([event.cycle_time.total_seconds() for event in self.__reference_model.data]),
        )

        LOGGER.verbose(
            "running time distribution is %s",
            scipy.stats.describe([event.cycle_time.total_seconds() for event in self.__running_model.data]),
        )

        # If models are not statistically equal, there is a drift
        if not self.__reference_model.statistically_equals(self.__running_model):
            # At the beginning all drifts are created as warnings
            drift = Drift(
                level=DriftLevel.WARNING,
                reference_model=self.__reference_model,
                running_model=self.__running_model,
            )

            # The first warning is the same as in the previous warning, or the warning itself if no previous warnings
            if self.__drift_warnings[-1].level == DriftLevel.WARNING:
                drift.first_warning = self.__drift_warnings[-1].first_warning
            else:
                drift.first_warning = deepcopy(drift)

            # Store the drift warning
            self.__drift_warnings.append(drift)

            LOGGER.verbose(
                "drift warning in the cycle time between reference model (%s - %s) and running model (%s - %s)",
                self.__reference_model.start, self.__reference_model.end,
                self.__running_model.start, self.__running_model.end)

            # When enough successive WARNINGS are found, the drift is confirmed
            if all(dft.level == DriftLevel.WARNING for dft in self.__drift_warnings):
                LOGGER.verbose(
                    "drift confirmed between reference model (%s - %s) and running model (%s - %s)",
                    self.__reference_model.start, self.__reference_model.end, self.__running_model.start, self.__running_model.end,
                )
                drift = Drift(
                    level=DriftLevel.CONFIRMED,
                    reference_model=self.__reference_model,
                    running_model=self.__running_model,
                    first_warning=self.__drift_warnings[-1].first_warning,
                )
                # when the drift is confirmed, the detector is restarted
                self.__reference_model = None
                self.__running_model = None
                self.__drift_warnings = deque([NO_DRIFT] * self.__warnings_to_confirm, maxlen=self.__warnings_to_confirm)
            # Return the drift
            return drift

        # If the model does not present a drift add a NONE
        LOGGER.verbose(
            "no drift between reference timeframe (%s - %s) and running timeframe (%s, %s)",
            self.__reference_model.start, self.__reference_model.end, self.__running_model.start, self.__running_model.end,
        )
        self.__drift_warnings.append(NO_DRIFT)
        return NO_DRIFT

    def update(self: typing.Self, event: Event) -> Drift:
        """
        Update the model with a new event and check if it presents a drift.

        If the event lies in the reference time window it will be added to both the reference and the running model.
        Otherwise, only the running model will be updated.

        Parameters
        ----------
        * `event`: *the new event to be added to the model*
        """
        # Initialize models if needed
        if self.__reference_model is None or self.__running_model is None:
            self.__initialize_models(event.enabled)
        # Drop the event if it is part of the warm-up period
        if event.enabled < self.__reference_model.start:
            LOGGER.spam("dropping warm-up event %r", event)
            return NO_DRIFT  # drop event, nothing more to do here
        # If the event is part of the reference model, update it
        if self.__reference_model.envelopes(event):
            LOGGER.spam("updating reference model (%s - %s) with event %r", self.__reference_model.start, self.__reference_model.end, event)
            self.__update_reference_model(event)
        # If the event is part of the running model, update it
        if self.__running_model.envelopes(event):
            LOGGER.spam("updating running model (%s - %s) with event %r", self.__reference_model.start, self.__reference_model.end, event)
            self.__update_running_model(event)
        # If the running model is complete (i.e., the event ends after the model end), update drifts and timeframes if needed
        if self.__running_model.completed(event.end):
            # Check for the presence of drifts
            LOGGER.verbose(
                "checking drift between reference model (%s - %s) and running model (%s - %s)",
                self.__reference_model.start, self.__reference_model.end, self.__running_model.start, self.__running_model.end,
            )
            drift = self.__update_drifts()

            # If no drift is confirmed, update the running model timeframe
            if drift.level != DriftLevel.CONFIRMED:
                # Update the timeframe as many times as needed for it to contain the event
                while self.__running_model.completed(event.end):
                    LOGGER.debug(
                        "updating running model to (%s - %s)",
                        self.__running_model.end - self.__overlap, self.__running_model.end - self.__overlap + self.__timeframe_size,
                    )
                    self.__running_model.update_timeframe(self.__running_model.end - self.__overlap, self.__timeframe_size)
            # Once drifts are checked and timeframes updated, we can recursively call the method with the same event again so it is added
            self.update(event)

            return drift

        return NO_DRIFT
