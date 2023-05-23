from __future__ import annotations

import time
from datetime import timedelta

from expert.model import *
from expert.drift import detect_drift
from expert.drift.causality import explain_drift
from expert.drift.features import DriftFeatures
from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.logger import LOGGER, Level, setup_logger
from expert.output import plot_features, print_causes
from expert.utils.concurrency import HeuristicsConcurrencyOracle
from expert.utils.log import infer_final_activities, infer_initial_activities


def __preprocess(event_log: Log) -> Log:
    event_log = list(event_log)

    for event in event_log:
        if event.resource == "Fake_Resource":
            event.resource = None

    return HeuristicsConcurrencyOracle(event_log).compute_enablement_timestamps()



if __name__ == "__main__":
    start = time.perf_counter_ns()

    setup_logger(Level.VERBOSE, "output.log")

    files = (
        ("log1", "../data/base_sim_scenario 1.csv"),
        ("log2", "../data/base_sim_scenario 2.csv"),
    )

    log = list(
        read_and_merge_csv_logs(
            files,
            attribute_mapping=EventMapping(start="start", end="end", enablement=None, case="case", activity="activity", resource="resource"),
            preprocessor=__preprocess,
        ),
    )

    detector = detect_drift(
        log=(event for event in log),
        timeframe_size=timedelta(days=30),
        overlap_between_models=timedelta(days=15),
        warm_up=timedelta(days=90),
        warnings_to_confirm=3,
        initial_activities=infer_initial_activities(log),
        final_activities=infer_final_activities(log),
    )

    for index, drift in enumerate(detector):
        features = DriftFeatures(drift)
        # plots = plot_features(features)
        # plots.savefig(f"causes-drift-{index}.svg")
        # causes = explain_drift(features)
        # print_causes(causes)

    end = time.perf_counter_ns()

    LOGGER.success("execution took %s", timedelta(microseconds=(end - start)/1000))
