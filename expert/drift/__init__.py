"""This module contains the functions needed for detecting and explaining performance drifts in a process model."""
from __future__ import annotations

import logging
import typing
from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil
from pprint import pprint

import numpy as np
import pandas as pd
from intervaltree import Interval, IntervalTree
from river.drift import ADWIN
from scipy.stats import cramervonmises_2samp as cramer_von_mises_test

from expert.drift.model import Model, Pair, Result
from expert.model import Event
from expert.utils import find_log_end, find_log_start
from expert.utils.filters import has_any_resource


def detect_drift(
        log: typing.Generator[Event, typing.Any, typing.Any],
        timeframe_size: timedelta,
        *,
        initial_activity: str = "START",
        final_activity: str = "END",
        alpha: float = 0.005,
        filters: typing.Iterable[typing.Callable[[Event], bool]] = (),
) -> Result | None:
    """Find and explain drifts in the performance of a process execution by monitoring its cycle time.

    The detection follows these steps:
    1. A reference model is built using the first events from the log. This model will be used later as the ground
       truth to check for the change causes.
    2. A running model is defined, using the last events from the log. This is the model that will be checked for
       changes.
    3. Events are read one by one, and the running model is updated, computing the cycle time for each completed
       case. This cycle time is monitored using a traditional concept drift algorithm, namely an *ADWIN*.
    4. If a change in the process cycle time is detected by the *ADWIN*, the causes leading to that change are analyzed

    Parameters
    ----------
    * `log`:                *the input event log*
    * `timeframe_size`:     *the size of the timeframe for the reference and drifting models*
    * `initial_activity`:   *the first activity of the process/process fragment to be monitored*
    * `final_activity`:     *the last activity of the process/process fragment to be monitored*
    * `alpha`:              *the sensitivity for the statistical scripts*
    * `filters`:            *a collection of filters that will be applied to the events before processing them*

    Returns
    -------
    * the detection result, with the point where the drift happened and the reasons for that drift
    """
    drift_detector = ADWIN(delta=alpha)

    # Create the model with the given parameters
    drift_model = Model(
        timeframe_size=timeframe_size,
        initial_activity=initial_activity,
        final_activity=final_activity,
    )

    # Iterate over the events in the log
    for index, event in enumerate(log):
        # Discard the event if it does not satisfy any of the conditions defined in the filters
        if any(not event_filter(event) for event_filter in filters):
            continue

        # Update the model with the new event
        case_duration_in_seconds = drift_model.update(event)

        # If the model update returned a time measurement, add it to the drift detector instance
        if case_duration_in_seconds is not None:
            drift_detector.update(case_duration_in_seconds)

        # If a change is detected, save the event and reset the model and the drift detector
        if drift_detector.drift_detected:
            logging.info("drift detected at event %(index)d: %(event)r", {"index": index, "event": event})
            # Return the drift with the causes for the change
            return explain_drift(drift_model, alpha=alpha)

    return None


def compute_arrival_rate(log: typing.Iterable[Event],
                         timeunit: timedelta = timedelta(minutes=1)) -> typing.Mapping[str, float]:
    """Compute the arrival rate for each activity in the log.

    To compute the arrival rate, the log timeframe is split in slots of size `timeunit` and, for each of these slots,
    we count how many events started within the slot timeframe. Finally, the count of events per slot is reduced by a
    mean, so we have the mean arrival rate per activity.

    Parameters
    ----------
    * `log`:        *an event log*
    * `timeunit`:   *the granularity used to compute the arrival rate*

    Returns
    -------
    * a mapping with pairs (`activity`, `arrival rate`)
    """
    # Get the set of activities
    activities = {event.activity for event in log}

    # Build time slots where arrival rate will be checked
    log_start: datetime = find_log_start(log)
    log_end: datetime = find_log_end(log)

    intervals = pd.interval_range(start=log_start, periods=ceil((log_end - log_start) / timeunit), freq=timeunit).array
    slots: list[Interval] = [Interval(interval.left, interval.right) for interval in intervals]

    # Build a timetable with all the event start timestamps for each activity
    event_timetable = defaultdict(IntervalTree)
    for event in log:
        event_timetable[event.activity][event.start:(event.start + timedelta(microseconds=1))] = event

    activity_rates = {}
    # Compute activity arrival rates
    for activity in activities:
        events_per_slot = []
        for slot in slots:
            # Get the events with the given activity that started in the time slot
            events_in_slot = len(event_timetable[activity][slot.begin:slot.end])

            events_per_slot.append(events_in_slot)
            logging.debug("activity %(activity)s slot %(slot)r events %(events)d",
                          {"activity": activity, "slot": slot, "events": events_in_slot})

        # Summarize the results for every activity computing the average arrival rate per time slot
        activity_rates[activity] = float(np.mean(np.array(events_per_slot)))

    return {key: activity_rates[key] for key in sorted(activity_rates.keys())}


def compute_resources_utilization_rate(log: typing.Iterable[Event],
                                       timeunit: timedelta = timedelta(minutes=1)) -> typing.Mapping[str, float]:
    """Compute the mean resource utilization for each resource in the event log.

    To perform this computation, the log timeframe is split in slots of size `timeunit` and, for each of these slots,
    the rate of occupation is computed by intersecting it with the events executed by each resource. Finally, the mean
    utilization rate for each resource is computed with the mean of the resource utilization for every time slot.

    Parameters
    ----------
    * `log`:        *an event log*
    * `timeunit`:   *the granularity used to compute the resources utilization rate

    Returns
    -------
    * a dictionary with pairs (`resource`, `utilization`) where usage is in the range [0.0, 1.0], being 0.0 no activity
      at all and 1.0 a fully used resource
    """
    # Filter events without resource
    filtered_log: list[Event] = [event for event in log if has_any_resource(event)]
    # Get the set of resources
    resources: set[str] = {event.resource for event in filtered_log}

    # Build time slots where frequency will be checked
    log_start: datetime = find_log_start(log)
    log_end: datetime = find_log_end(log)

    intervals = pd.interval_range(start=log_start, periods=ceil((log_end - log_start) / timeunit), freq=timeunit).array
    slots: list[Interval] = [Interval(interval.left, interval.right) for interval in intervals]

    # Build resources timetables
    resources_timetable: defaultdict[str, IntervalTree] = defaultdict(IntervalTree)
    for event in filtered_log:
        resources_timetable[event.resource][event.start : event.end] = event

    # Compute resource occupancy for each resource
    resources_occupancy: defaultdict[str, list[float]] = defaultdict(list)
    for resource in resources:
        for slot in slots:
            # Get the events in the time slot for the resource
            events_in_slot = resources_timetable[resource][slot.begin:slot.end]
            # Compute the percentage of the slot time that is used by the events
            used_time_in_slot = sum(
                ((min(event.end, slot.end) - max(event.begin, slot.begin)) for event in events_in_slot),
                start=timedelta(),
            ) / (slot.end - slot.begin)
            resources_occupancy[resource].append(used_time_in_slot)
            logging.debug("resource %(resource)s slot %(slot)r usage %(usage)06f",
                          {"resource": resource, "slot": slot, "usage": used_time_in_slot})

    # Summarize the results computing the average resource usage for every time slot
    return {
        key: float(np.mean(np.array(resources_occupancy[key])))
        for key in sorted(resources_occupancy.keys())
    }



def compute_waiting_times(log: typing.Iterable[Event]) -> typing.Mapping[str, timedelta]:
    """Compute the average waiting time for each activity in the log.

    To compute the average waiting time events are grouped by their activity and then the waiting times per event are
    aggregated by the mean.

    Parameters
    ----------
    * `log`:    *an event log*

    Returns
    -------
    * a dictionary with pairs (`activity`, `average waiting time`)
    """
    # get the set of activities
    activities = {event.activity for event in log}
    # group events by activity and store the waiting time for each event
    waiting_times = {
        activity: [event.waiting_time for event in log if event.activity == activity] for activity in activities
    }
    # aggregate events waiting time by the average
    return {
        activity: sum(waiting_times[activity], timedelta(0)) / len(waiting_times[activity]) for activity in activities
    }

def check_arrival_rate(drift_model: Model, timeframe: timedelta, *, alpha: float = 0.05) -> bool:
    """Check whether the arrival rate for the activities did change between the reference and the running model.

    To check if there is a statistically significant change, the distribution of arrival rates are compared using a
    Cramer von Mises statistical test. A change is detected if the p-value for comparing if the two distributions are
    the same falls below the `alpha` value.

    Parameters
    ----------
    * `model`:      *a drift model containing the sublogs being compared*
    * `timeunit`:   *the granularity for checking the arrival rate of the activities*
    * `alpha`:      *the significance value for the statistical test*

    Returns
    -------
    * a boolean indicating whether the model presents a change in the arrival rates
    """
    # Compute the arrival rate for the reference and the running models
    reference_arrival_rate = compute_arrival_rate(drift_model.reference_model, timeframe)
    running_arrival_rate = compute_arrival_rate(drift_model.running_model, timeframe)

    # Perform the statistical test to look for changes
    test_result = cramer_von_mises_test(list(running_arrival_rate.values()),
                                        list(reference_arrival_rate.values()))

    logging.debug("reference arrival rate: %(reference_rate)r", {"reference_rate": reference_arrival_rate})
    logging.debug("running arrival rate: %(running_rate)r", {"running_rate": running_arrival_rate})
    logging.debug("test result for arrival rates: %(pvalue)f", {"pvalue": test_result.pvalue})

    return test_result.pvalue <= alpha


def check_resources_utilization(drift_model: Model, timeframe: timedelta, *, alpha: float = 0.05) -> bool:
    """Check whether the resource utilization rates did change between the reference and the running model.

    To perform this checking, the average utilization rate per resource is computed for both the reference and
    the running models, and the distributions of utilization rates are compared using a statistical test.
    A change in the distributions is detected when the p-value for the test falls below the given threshold `alpha`.

    Parameters
    ----------
    * `model`:      *a drift model containing the sublogs being compared*
    * `timeunit`:   *the granularity for computing the resource utilization rate*
    * `alpha`:      *the significance value for the statistical test*

    Returns
    -------
    * a boolean indicating whether the model presents a change in the resources utilization rates or not
    """
    # Compute the resource utilization for the reference and the running models
    reference_resources_utilization = compute_resources_utilization_rate(drift_model.reference_model, timeframe)
    running_resources_utilization = compute_resources_utilization_rate(drift_model.running_model, timeframe)

    # Perform the statistical test to look for changes
    test_result = cramer_von_mises_test(list(reference_resources_utilization.values()),
                                        list(running_resources_utilization.values()))

    logging.debug("reference resource utilization: %(reference_usage)r",
                  {"reference_usage": reference_resources_utilization})
    logging.debug("running resource utilization: %(running_usage)r", {"running_usage": running_resources_utilization})
    logging.debug("test result for resources utilization rates: %(pvalue)f", {"pvalue": test_result.pvalue})

    return test_result.pvalue < alpha



def explain_drift(drift_model: Model, *, alpha: float = 0.005) -> Result:
    """
    Find the actionable causes of a drift given the drift model.

    This function computes multiple metrics for the reference and running models and, based on that metrics, it provides
    insights about actionable causes for the drift, mainly related to the resources.


    Parameters
    ----------
    * `drift_model`:    *the model containing the running and reference events for the drift*
    * `alpha`:          *the threshold for the confidence of the statistical tests used to determine the presence of a
                         change*

    Returns
    -------
    * the result of the drift detection with the drift causes
    """
    # Compute the different metrics
    reference_arrival_rate = compute_arrival_rate(drift_model.reference_model, timeunit=timedelta(minutes=1))
    running_arrival_rate = compute_arrival_rate(drift_model.running_model, timeunit=timedelta(minutes=1))
    reference_resource_utilization_rate = compute_resources_utilization_rate(drift_model.reference_model,
                                                                             timeunit=timedelta(minutes=1))
    running_resource_utilization_rate = compute_resources_utilization_rate(drift_model.running_model,
                                                                           timeunit=timedelta(minutes=1))
    reference_waiting_times = compute_waiting_times(drift_model.reference_model)
    running_waiting_times = compute_waiting_times(drift_model.running_model)

    # Build the result using the values previously computed
    result = Result(
        model=Pair(
            reference=drift_model.reference_model,
            running=drift_model.running_model,
        ),
        arrival_rate=Pair(
            reference=reference_arrival_rate,
            running=running_arrival_rate,
        ),
        resource_utilization_rate=Pair(
            reference=reference_resource_utilization_rate,
            running=running_resource_utilization_rate,
        ),
        waiting_time=Pair(
            reference=reference_waiting_times,
            running=running_waiting_times,
        ),
    )

    check_arrival_rate(drift_model, timedelta(days=5), alpha=alpha)
    check_resources_utilization(drift_model, timedelta(days=5), alpha=alpha)

    pprint(f"reference waiting times: {reference_waiting_times}")
    pprint(f"running waiting times: {running_waiting_times}")
    pprint(f"reference arrival rate: {reference_arrival_rate}")
    pprint(f"running arrival rate: {running_arrival_rate}")
    pprint(f"reference resource utilization rate: {reference_resource_utilization_rate}")
    pprint(f"running resource utilization rate: {running_resource_utilization_rate}")
    return result
