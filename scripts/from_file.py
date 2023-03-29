from __future__ import annotations

import time
from datetime import timedelta

from expert.drift import detect_drift
from expert.drift.causes import explain_drift
from expert.drift.model import DriftCauses
from expert.drift.plot import plot_causes
from expert.input.csv import DEFAULT_APROMORE_CSV_MAPPING, read_and_merge_csv_logs
from expert.logger import LOGGER, Level, setup_logger
from expert.utils import (
    compute_average_case_duration,
    compute_average_inter_case_time,
    infer_final_activities,
    infer_initial_activities,
)
from expert.utils.concurrency import HeuristicsConcurrencyOracle


def __print_causes(_causes: DriftCauses) -> None:
    LOGGER.notice("drift causes:")
    LOGGER.notice("    execution times changed: %s", _causes.execution_time_changed)
    LOGGER.notice("    waiting times changed: %s", _causes.waiting_time_changed)
    LOGGER.notice("    arrival rate changed: %s", _causes.arrival_rate_changed)
    LOGGER.notice("    resource utilization rate changed: %s", _causes.resource_utilization_rate_changed)

if __name__ == '__main__':
    start = time.perf_counter_ns()

    setup_logger(Level.INFO)

    files = (
        ("log1", "../data/simple/base.csv"),
    )

    log = read_and_merge_csv_logs(
        files,
        attribute_mapping=DEFAULT_APROMORE_CSV_MAPPING,
        preprocessor=lambda evt_log: HeuristicsConcurrencyOracle(evt_log).compute_enablement_timestamps(),
    )

    num_cases = 200

    log = list(log)
    initial_activities = infer_initial_activities(log)
    final_activities = infer_final_activities(log)
    avg_case_duration = compute_average_case_duration(log)
    time_between_cases = compute_average_inter_case_time(log, initial_activities)

    detector = detect_drift(
        log=(event for event in log),
        timeframe_size=num_cases * time_between_cases + avg_case_duration,
        overlap_between_models=(num_cases * time_between_cases + avg_case_duration) / 2,
        warm_up=num_cases * time_between_cases,
        warnings_to_confirm=3,
        initial_activities=initial_activities,
        final_activities=final_activities,
    )

    for index, drift in enumerate(detector):
        causes = explain_drift(drift)
        __print_causes(causes)
        plots = plot_causes(causes)
        plots.savefig(f"causes-drift-{index}.svg")

    end = time.perf_counter_ns()

    LOGGER.success("execution took %s", timedelta(microseconds=(end - start)/1000))
