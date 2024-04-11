from __future__ import annotations

from expert.drift import detect_drift
from expert.drift.causality import explain_drift
from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.logger import LOGGER, Level, setup_logger
from expert.model import *
from expert.output import print_causes
from expert.timer import DEFAULT_TIMER as TIMER, profile


@profile()
def __preprocess(event_log: Log) -> Log:

    event_log = list(event_log)

    # Remove fake resources
    for event in event_log:
        if event.resource is None:
            event.resource = "__UNKNOWN__"

    return event_log


if __name__ == "__main__":
    with TIMER.profile(__name__):
        setup_logger(Level.NOTICE, destination="output.log")

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
            ),
        )

        # 1 year  -> 150k events -> 01:30
        # 6 month ->  75k events -> 00:50
        # 3 month ->  35k events -> 00:30
        # 1 month ->  10k events -> 00:20
        # 1 week  ->   3k events -> 00:15
        # 1 day   ->  400 events -> 00:15

        detector = detect_drift(
            log=log,
            timeframe_size=timedelta(days=7),
            overlap_between_models=timedelta(days=0),
            warm_up=timedelta(days=7),
            warnings_to_confirm=3,
        )

        for index, drift in enumerate(detector):
            causes = explain_drift(drift)
            print_causes(causes)
        # 
        #     case_plots = plot_case_features(case_features)
        #     activity_plots = plot_activity_features(activity_features)
        #     case_plots.savefig(f"drift-{index}.cases.svg")
        #     activity_plots.savefig(f"drift-{index}.activities.svg")
        # 
        #     export_causes(causes, filename=f"drift-{index}.tree.json")
        #     export_causes(causes, filename=f"drift-{index}.tree.yaml")

    LOGGER.success("execution took %s", TIMER.elapsed(__name__))

    LOGGER.success("detailed timings:")
    for line in str(TIMER).splitlines():
        LOGGER.success(f"    {line}")
    