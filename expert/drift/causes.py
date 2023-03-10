import logging
import typing
from datetime import timedelta

from expert.drift import Model
from expert.drift.model import Pair, Result
from expert.utils import (
    compute_activity_arrival_rate,
    compute_activity_waiting_times,
    compute_resources_utilization_rate,
)
from expert.utils.statistical_tests import ks_test


def check_arrival_rate(
        drift_model: Model,
        timeframe: timedelta,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool],
) -> bool:
    """Check whether the arrival rate for the activities did change between the reference and the running model.

    To check if there is a statistically significant change, the distribution of arrival rates are compared using a
    statistical test.

    Parameters
    ----------
    * `model`:      *a drift model containing the sublogs being compared*
    * `timeunit`:   *the granularity for checking the arrival rate of the activities*
    * `test`:       *the test for evaluating if there are any difference between the reference and the running
                     models*

    Returns
    -------
    * a boolean indicating whether the model presents a change in the arrival rates
    """
    # Compute the arrival rate for the reference and the running models
    reference_arrival_rate = compute_activity_arrival_rate(drift_model.reference_model, timeframe)
    running_arrival_rate = compute_activity_arrival_rate(drift_model.running_model, timeframe)

    logging.debug("reference arrival rate: %(reference_rate)r", {"reference_rate": reference_arrival_rate})
    logging.debug("running arrival rate: %(running_rate)r", {"running_rate": running_arrival_rate})

    # Check if there are changes
    return test(list(running_arrival_rate.values()), list(reference_arrival_rate.values()))


def check_resources_utilization(
        drift_model: Model,
        timeframe: timedelta,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool],
) -> bool:
    """Check whether the resource utilization rates did change between the reference and the running model.

    To perform this checking, the average utilization rate per resource is computed for both the reference and
    the running models, and the distributions of utilization rates are compared using a statistical test.

    Parameters
    ----------
    * `model`:      *a drift model containing the sublogs being compared*
    * `timeunit`:   *the granularity for computing the resource utilization rate*
    * `test`:       *the test for evaluating if there are any difference between the reference and the running
                     models*

    Returns
    -------
    * a boolean indicating whether the model presents a change in the resources utilization rates or not
    """
    # Compute the resource utilization for the reference and the running models
    reference_resources_utilization = compute_resources_utilization_rate(drift_model.reference_model, timeframe)
    running_resources_utilization = compute_resources_utilization_rate(drift_model.running_model, timeframe)

    logging.debug("reference resource utilization: %(reference_usage)r",
                  {"reference_usage": reference_resources_utilization})
    logging.debug("running resource utilization: %(running_usage)r", {"running_usage": running_resources_utilization})

    # Check if there are changes
    return test(list(reference_resources_utilization.values()), list(running_resources_utilization.values()))


def check_waiting_times(
        drift_model: Model,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool],
) -> bool:
    """Check whether the waiting time for each activity changed between the reference and running models.

    To perform this checking, the average waiting time per activity is computed for both the reference and
    the running models, and the distributions of waiting times are compared using a statistical test.

    Parameters
    ----------
    * `model`:      *a drift model containing the sublogs being compared*
    * `timeunit`:   *the granularity for computing the resource utilization rate*
    * `test`:       *the test for evaluating if there are any difference between the reference and the running
                     models*

    Returns
    -------
    * a boolean indicating whether the model presents a change in the resources utilization rates or not
    """
    reference_waiting_times = compute_activity_waiting_times(drift_model.reference_model)
    running_waiting_times = compute_activity_waiting_times(drift_model.running_model)

    logging.debug("reference waiting times: %(reference_waiting_times)r", {"reference_waiting_times": reference_waiting_times})
    logging.debug("running waiting times: %(running_waiting_times)r", {"running_waiting_times": running_waiting_times})

    # Check if there are changes
    return test(
        [value.total_seconds() for value in reference_waiting_times.values()],
        [value.total_seconds() for value in running_waiting_times.values()],
    )

def explain_drift(
        drift_model: Model,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool] = ks_test(),
) -> Result:
    """
    Find the actionable causes of a drift given the drift model.

    This function computes multiple metrics for the reference and running models and, based on that metrics, it provides
    insights about actionable causes for the drift, mainly related to the resources.


    Parameters
    ----------
    * `drift_model`:    *the model containing the running and reference events for the drift*
    * `test`:           *the test for evaluating if there are any difference between the reference and the running
                        models*

    Returns
    -------
    * the result of the drift detection with the drift causes
    """
    # Compute the different metrics
    reference_arrival_rate = compute_activity_arrival_rate(drift_model.reference_model, timeunit=timedelta(days=1))
    running_arrival_rate = compute_activity_arrival_rate(drift_model.running_model, timeunit=timedelta(days=1))

    reference_resource_utilization_rate = compute_resources_utilization_rate(drift_model.reference_model,
                                                                               timeunit=timedelta(days=1))
    running_resource_utilization_rate = compute_resources_utilization_rate(drift_model.running_model,
                                                                             timeunit=timedelta(days=1))

    reference_waiting_times = compute_activity_waiting_times(drift_model.reference_model)
    running_waiting_times = compute_activity_waiting_times(drift_model.running_model)

    # Build the result using the values previously computed
    return Result(
        model=Pair(
            reference=drift_model.reference_model,
            running=drift_model.running_model,
        ),
        case_duration=Pair(
            reference=drift_model.reference_model_durations,
            running=drift_model.running_model_durations,
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
        arrival_rate_changed=check_arrival_rate(drift_model, timedelta(days=1), test=test),
        resource_utilization_rate_changed=check_resources_utilization(drift_model, timedelta(days=1), test=test),
        waiting_time_changed=check_waiting_times(drift_model, test=test),
    )
