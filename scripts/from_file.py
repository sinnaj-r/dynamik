from __future__ import annotations

import json
import time
from pprint import pprint

from expert.drift import detect_drift
from expert.drift.causality import explain_drift
from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.logger import LOGGER, Level, setup_logger
from expert.model import *
from expert.output import print_causes, export_causes, plot_case_features, plot_activity_features
from expert.utils.log import infer_final_activities, infer_initial_activities


def __preprocess(event_log: Log) -> Log:

    event_log = list(event_log)

    # Remove fake resources
    for event in event_log:
        if event.resource is None:
            event.resource = "__UNKNOWN__"

    return event_log


if __name__ == "__main__":
    start = time.perf_counter_ns()

    setup_logger(Level.NOTICE, "output.log")

    files = (
        # ("work_orders", "../data/logs/real/work_orders.csv"),
        ("base", "../data/logs/base-sequence.csv"),
        # ("no-available-on-night", "../data/logs/base-sequence-with-unavailability.csv"),
        # ("fast-exc", "../data/logs/base-sequence-fast-execution.csv"),
        # ("long-run", "../data/logs/base-sequence-long-run.csv"),
        # ("with-idle-time", "../data/logs/base-sequence-long-run-limited-avail-no-wait.csv"),
        ("batching", "../data/logs/base-sequence-batching.csv"),
    )

    log = list(
        read_and_merge_csv_logs(
            files,
            attribute_mapping=EventMapping(start="start_time", end="end_time", enablement="enable_time", case="case_id", activity="activity", resource="resource"),
            preprocessor=__preprocess,
            add_artificial_start_end_events=True
        ),
    )

    initial = infer_initial_activities(log)
    final = infer_final_activities(log)

    detector = detect_drift(
        log=(event for event in log),
        timeframe_size=timedelta(days=7),
        overlap_between_models=timedelta(days=2),
        warm_up=timedelta(days=7),
        warnings_to_confirm=3,
        initial_activities=initial,
        final_activities=final,
    )

    for index, drift in enumerate(detector):
        case_features = drift.case_features
        activity_features = drift.activity_features

        causes = explain_drift(drift)
        print_causes(causes)

        case_plots = plot_case_features(case_features)
        activity_plots = plot_activity_features(activity_features)
        case_plots.savefig(f"drift-{index}.cases.svg")
        activity_plots.savefig(f"drift-{index}.activities.svg")

        export_causes(causes, filename=f"drift-{index}.tree.json")
        export_causes(causes, filename=f"drift-{index}.tree.yaml")

    end = time.perf_counter_ns()

    LOGGER.success("execution took %s", timedelta(microseconds=(end - start)/1000))
    