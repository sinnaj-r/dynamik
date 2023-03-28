"""This module contains the logic for evaluating the actionable causes of a drift in the cycle time of a process"""
import typing
from datetime import timedelta

import scipy

from expert.drift.model import Drift, DriftCauses
from expert.drift.model import _Pair as Pair
from expert.utils import (
    compute_activity_arrival_rate,
    compute_activity_execution_times,
    compute_activity_waiting_times,
    compute_resources_utilization_rate,
)


def __default_cause_test_factory(alpha: float = 0.05) -> \
        typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool]:
    def __test(reference: typing.Iterable[float], running: typing.Iterable[float]) -> bool:
        return scipy.stats.mannwhitneyu(list(reference), list(running)).pvalue < alpha
    return __test


def __check_drift(
        reference_data: typing.Mapping[str, typing.Iterable[float]],
        running_data: typing.Mapping[str, typing.Iterable[float]],
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool],
) -> typing.Mapping[str, bool]:
    # Perform the statistical test over each pair of items in the given mappings to check if there has been a change
    return {
        key: test(reference_data[key], running_data[key]) if key in reference_data and key in running_data else True
        for key in set(list(reference_data.keys()) + list(running_data.keys()))
    }


def explain_drift(
        drift_model: Drift,
        *,
        test: typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool] = __default_cause_test_factory(),
        granularity: timedelta = timedelta(hours=1),
) -> DriftCauses:
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
    reference_arrival_rate = compute_activity_arrival_rate(
        drift_model.reference_model,
        timeunit=granularity,
    )
    running_arrival_rate = compute_activity_arrival_rate(
        drift_model.running_model,
        timeunit=granularity,
    )

    reference_resource_utilization_rate = compute_resources_utilization_rate(
        drift_model.reference_model,
        timeunit=granularity,
    )
    running_resource_utilization_rate = compute_resources_utilization_rate(
        drift_model.running_model,
        timeunit=granularity,
    )

    reference_waiting_times = compute_activity_waiting_times(drift_model.reference_model)
    running_waiting_times = compute_activity_waiting_times(drift_model.running_model)

    reference_execution_times = compute_activity_execution_times(drift_model.reference_model)
    running_execution_times = compute_activity_execution_times(drift_model.running_model)

    # Build the result using the values previously computed
    return DriftCauses(
        model=Pair(
            reference=drift_model.reference_model,
            running=drift_model.running_model,
        ),
        case_duration=Pair(
            reference=drift_model.reference_durations,
            running=drift_model.running_durations,
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
        execution_time=Pair(
            reference=reference_execution_times,
            running=running_execution_times,
        ),
        arrival_rate_changed=any(
            __check_drift(
                reference_arrival_rate,
                running_arrival_rate,
                test=test,
            ).values(),
        ),
        resource_utilization_rate_changed=any(
            __check_drift(
                reference_resource_utilization_rate,
                running_resource_utilization_rate,
                test=test,
            ).values(),
        ),
        waiting_time_changed=any(
            __check_drift(
                {
                    key: [value.total_seconds() for value in reference_waiting_times[key]]
                    for key in reference_waiting_times
                },
                {
                    key: [value.total_seconds() for value in running_waiting_times[key]]
                    for key in running_waiting_times
                },
                test=test,
            ).values(),
        ),
        execution_time_changed=any(
            __check_drift(
                {
                    key: [value.total_seconds() for value in reference_execution_times[key]]
                    for key in reference_execution_times
                },
                {
                    key: [value.total_seconds() for value in running_execution_times[key]]
                    for key in running_execution_times
                },
                test=test,
            ).values(),
        ),
    )


