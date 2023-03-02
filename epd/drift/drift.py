from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from math import ceil

import numpy as np
import pandas as pd
from intervaltree import Interval, IntervalTree
from river.drift import ADWIN
from scipy.stats import cramervonmises_2samp as cramer_von_mises_test

from epd.model import Event
from .model import Model


def detect_drift(
        log: Iterable[Event],
        timeframe_size: timedelta,
        *,
        initial_activity: str = "START",
        final_activity: str = "END",
        alpha: float = 0.005,
) -> Iterable[Event]:
    """
    Find and explain drifts in the performance of a process.

    This detection follows the next steps:
        1. A reference model is built using the first events from the log. This model will be used later as the ground
         truth to check for the change causes.
        2. A running model is defined, using the last events from the log. This is the model that will be checked for
           changes.
        3. Events are read one by one, and the running model is updated, computing the cycle time for each completed
           case. This cycle time is monitored using a traditional concept drift algorithm, namely an ADWIN.
        4. If a change in the process cycle time is detected by the ADWIN, we analyze the causes leading to that change

    :param log: the input event log containing start and finish timestamps for each event
    :param timeframe_size: the size of the timeframe for the reference and drifting models
    :param initial_activity: the first activity of the process/process fragment to be monitored
    :param final_activity: the last activity of the process/process fragment to be monitored
    :param alpha: the sensitivity for the statistical scripts

    :return: a list of events causing drift.
    """
    drifts = []
    drift_detector = ADWIN(delta=alpha)

    # Create the model with the given parameters
    model = Model(
        timeframe_size=timeframe_size,
        initial_activity=initial_activity,
        final_activity=final_activity,
    )

    # Iterate over the events in the log
    for index, event in enumerate(log):
        # Update the model with the new event
        case_duration_in_seconds = model.update(event=event)

        # If the model update returned a time measurement, add it to the drift detector instance
        if case_duration_in_seconds is not None:
            drift_detector.update(case_duration_in_seconds)

        # If a change is detected, save the event and reset the model and the drift detector
        if drift_detector.drift_detected:
            drifts.append(event)
            logging.info("drift detected at event %(index)d: %(event)r", {"index": index, "event": event})
            # Find causality for the change in the cycle time
            find_causality(model, timeframe_size, alpha=alpha)
            # Reset models
            return drifts

    return drifts


def compute_arrival_rate(log: Iterable[Event], timeunit: timedelta = timedelta(minutes=1)) -> dict[str, float]:
    """
    Compute the arrival rate for each activity in the log.

    Computes the arrival rate for each activity in the given log. To compute the arrival rate the events are grouped by
    its activity name and then the count of events per activity are divided by the complete log timeframe and the
    timeunit, so the result is a dictionary with pairs (activity, mean arrivals per time unit)

    :param log: an event log
    :param timeunit: the granularity used to compute the arrival rates
    :return: a dictionary with pairs (activity, arrival rate)
    """
    # Transform the event insatances to dictionaries and build a new dataframe with them
    df_events = pd.DataFrame(evt.__dict__ for evt in log)
    # Compute the log timeframe as the interval [min date in log, max date in log]
    timeframe = max(df_events["start"].max(), df_events["end"].max()) \
                - min(df_events["start"].min(), df_events["end"].min())
    # Group events by their activity and count the number of executions
    activity_df = df_events.groupby(["activity"]).agg({"start": list})
    activity_df["count"] = activity_df["start"].apply(len)
    # Compute the arrival rate for each activity in the log timeframe and with the given time unit
    activity_df["rate"] = activity_df["count"] / (timeframe / timeunit)

    # Build the final dict with the arrival rates
    activity_rates = {}
    for activity, row in activity_df.iterrows():
        activity_rates[activity] = row["rate"]

    return activity_rates


def compute_resources_usage(log: Iterable[Event], timeunit: timedelta) -> dict[str, float]:
    """
    Compute the mean resource usage for each resource in the event log.

    Computes the resources utilization rate for each resource present in the given event log. To perform this
    computation, the log timeframe is split in slots of size `timeunit` and, for each of these slots, the percentage of
    occupation is computed by intersecting it with the events executed by each resource. Finally, the utilization rate
    for each resource is computed with the mean of the resource utilization for every time slot.

    :param log: an event log with resources information
    :param timeunit: the granularity used to compute resources usage
    :return: a dictionary with pairs resource->usage, where usage is in the range [0.0, 1.0], being 0.0 no activity at
             all and 1.0 a fully used resource
    """
    timeunit = timeunit if timeunit is not None else timedelta(minutes=1)

    # Filter artificial events from the log (duration = 0 or no resource)
    filtered_log = [event for event in log if
                    (event.end != event.start) and (event.resource != "NONE")]
    # Get the set of resources
    resources = {event.resource for event in filtered_log}

    # Build time slots where frequency will be checked
    log_start = min(event.start for event in log)
    log_end = max(event.end for event in log)
    slots = list(
        pd.interval_range(start=log_start, periods=ceil((log_end - log_start) / timeunit),
                          freq=timeunit).values,
    )

    # Build resources timetables
    resources_timetable = defaultdict(IntervalTree)
    for event in filtered_log:
        resources_timetable[event.resource].add(Interval(event.start, event.end, event))

    # Compute resource occupancy for each resource
    resources_occupancy = {}
    for resource in resources:
        per_slot_usage = []
        for slot in slots:
            # Get the events in the time slot for the resource
            events_in_slot = resources_timetable[resource][slot.left:slot.right]
            # Compute the percentage of the slot time that is used by the events
            used_time_in_slot = sum(
                (
                    (min(event.end, slot.right) - max(event.begin, slot.left))
                    for event in events_in_slot
                ),
                start=timedelta(),
            ) / (slot.right - slot.left)
            per_slot_usage.append(used_time_in_slot)
            logging.debug("resource %(resource)s slot %(slot)r usage %(usage)06f",
                          {"resource": resource, "slot": slot, "usage": used_time_in_slot})

        # Summarize the results computing the average resource usage for every time slot
        resources_occupancy[resource] = float(np.mean(np.array(per_slot_usage)))

    return resources_occupancy


def check_arrival_rate(model: Model, timeframe: timedelta, *, alpha: float = 0.05) -> bool:
    """
    Check whether the arrival rate for the activities did change between the reference and the running model.

    Checks if there has been a change in the arrival rate per activity between two snapshots of
    the log in two different moments. To check if there is a change, the distribution of arrival
    rates are compared using a Cramer von Mises statistical test. A change is detected if the
    p-value for comparing if the two distributions are the same falls below alpha.

    :param model: the model containing the two snapshots of the process log
    :param timeframe: the granularity for checking the arrival rate of the activities
    :param alpha: the threshold for the statistical test
    :return: a boolean indicating whether the model contains a change or not
    """
    # Compute the arrival rate for the reference and the running models
    reference_arrival_rate = compute_arrival_rate(model.reference_model, timeframe)
    running_arrival_rate = compute_arrival_rate(model.running_model, timeframe)

    # Perform the statistical test to look for changes
    test_result = cramer_von_mises_test(list(running_arrival_rate.values()),
                                        list(reference_arrival_rate.values()))

    logging.debug("reference arrival rate: %(reference_rate)r", {"reference_rate": reference_arrival_rate})
    logging.debug("running arrival rate: %(running_rate)r", {"running_rate": running_arrival_rate})
    logging.debug("test result for arrival rates: %(pvalue)f", {"pvalue": test_result.pvalue})

    print(f"Reference arrival rate:: {reference_arrival_rate!r}")
    print(f"Running arrival rate: {running_arrival_rate!r}")

    return test_result.pvalue <= alpha


def check_resources_usage(model: Model, timeframe: timedelta, *, alpha: float = 0.05) -> bool:
    """
    Check whether the resource usage did change between the reference and the running model.

    Checks if there has been any change in the resources utilization between two snapshots of the log.
    To perform this checking, the average utilization rate per resource is computed for both the reference and
    the running models, and the distributions of utilization rates are compared using a Cramer von Mises test.
    A change in the distributions is detected when the p-value for the test falls below the given threshold alpha.

    :param model: the model containing the two snapshots of the process log
    :param timeframe: the granularity for computing the resource usage rate
    :param alpha: the threshold for the statistical test
    :return: a boolean indicating whether the model contains a change or not
    """
    # Compute the resource usage for the reference and the running models
    reference_resources_usage = compute_resources_usage(model.reference_model, timeframe)
    running_resources_usage = compute_resources_usage(model.running_model, timeframe)

    # Perform the statistical test to look for changes
    test_result = cramer_von_mises_test(list(running_resources_usage.values()),
                                        list(reference_resources_usage.values()))

    logging.debug("reference resource utilization: %(reference_usage)r", {"reference_usage": reference_resources_usage})
    logging.debug("running resource utilization: %(running_usage)r", {"running_usage": running_resources_usage})
    logging.debug("test result for resources utilization rates: %(pvalue)f", {"pvalue": test_result.pvalue})

    print(f"Reference resource usage: {reference_resources_usage!r}")
    print(f"Running resource usage: {running_resources_usage!r}")

    return test_result.pvalue < alpha


def find_causality(model: Model, timeframe: timedelta, *, alpha: float = 0.005) -> None:
    arrival_rate_changed = check_arrival_rate(model, timeframe, alpha=alpha)
    resources_usage_changed = check_resources_usage(model, timedelta(days=5), alpha=alpha)

    print(f"Arrival rate changed: {arrival_rate_changed}")
    print(f"Resources usage changed: {resources_usage_changed}")
