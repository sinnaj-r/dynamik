from __future__ import annotations

import json
from datetime import timedelta

from expert.drift.detection import detect_drift
from expert.drift.causality import explain_drift
from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.output import print_causes, export_causes
from expert.utils.logger import LOGGER, Level, setup_logger
from expert.utils.timer import DEFAULT_TIMER as TIMER

if __name__ == "__main__":
    with TIMER.profile(__name__):
        setup_logger(Level.NOTICE, destination="output.log", disable_third_party_warnings=True)

        files = (
            # "../data/logs/real/work_orders.csv",
            "../data/logs/base-sequence.csv",
            # "../data/logs/base-sequence-with-unavailability.csv",
            # "../data/logs/base-sequence-fast-execution.csv",
            # "../data/logs/base-sequence-long-run.csv",
            # "../data/logs/base-sequence-long-run-limited-avail-no-wait.csv",
            "../data/logs/base-sequence-batching.csv",
        )

        log = read_and_merge_csv_logs(
            files,
            attribute_mapping=EventMapping(start="start_time", end="end_time", enablement="enable_time", case="case_id", activity="activity", resource="resource"),
        )

        detector = detect_drift(
            log=log,
            timeframe_size=timedelta(days=7),
            overlap_between_models=timedelta(days=0),
            warm_up=timedelta(days=7),
            warnings_to_confirm=3,
        )

        for index, drift in enumerate(detector):
            causes = explain_drift(drift, first_activity="Sequence A", last_activity="Sequence C")
            tree = export_causes(causes)
            print_causes(causes)
            with open(f"drift_{index}.json", "w") as file:
                json.dump(tree, file, indent=4)

    LOGGER.success("execution took %s", TIMER.elapsed(__name__))

    LOGGER.success("detailed timings:")
    for line in str(TIMER).splitlines():
        LOGGER.success(f"    {line}")
    