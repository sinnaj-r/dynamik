from __future__ import annotations

import time
import typing
from datetime import timedelta

from expert.drift import detect_drift
from expert.drift.causes import explain_drift
from expert.drift.model import DriftCauses
from expert.drift.plot import plot_causes
from expert.input.csv import DEFAULT_CSV_MAPPING, read_and_merge_csv_logs
from expert.logger import LOGGER, Level, setup_logger
from expert.model import Event
from expert.utils.concurrency import HeuristicsConcurrencyOracle
from expert.utils.log import infer_final_activities, infer_initial_activities


def __print_causes(_causes: DriftCauses) -> None:
    LOGGER.notice("drift causes:")
    LOGGER.notice("    execution times changed: %s", _causes.execution_time_changed)
    LOGGER.notice("    waiting times changed: %s", _causes.waiting_time_changed)
    LOGGER.notice("    arrival rate changed: %s", _causes.arrival_rate_changed)
    LOGGER.notice("    resource utilization rate changed: %s", _causes.resource_utilization_rate_changed)

def __preprocess(event_log: typing.Iterable[Event]) -> typing.Iterable[Event]:
    event_log = list(event_log)

    for event in event_log:
        if event.resource == "Fake_Resource":
            event.resource = None

    return HeuristicsConcurrencyOracle(event_log).compute_enablement_timestamps()

if __name__ == "__main__":
    start = time.perf_counter_ns()

    setup_logger(Level.VERBOSE)

    files = (
        ("batch", "../data/simple/batches.csv"),
    )

    log = list(
        read_and_merge_csv_logs(
            files,
            attribute_mapping=DEFAULT_CSV_MAPPING,
            preprocessor=__preprocess,
        ),
    )

    detector = detect_drift(
        log=(event for event in log),
        timeframe_size=timedelta(days=7),
        overlap_between_models=timedelta(days=7) / 2,
        warm_up=timedelta(days=7),
        warnings_to_confirm=5,
        initial_activities=infer_initial_activities(log),
        final_activities=infer_final_activities(log),
    )

    for index, drift in enumerate(detector):
        causes = explain_drift(drift)
        __print_causes(causes)
        plots = plot_causes(causes)
        plots.savefig(f"causes-drift-{index}.svg")

    end = time.perf_counter_ns()

    LOGGER.success("execution took %s", timedelta(microseconds=(end - start)/1000))
