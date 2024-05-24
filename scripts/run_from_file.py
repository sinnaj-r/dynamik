from __future__ import annotations

import json
from datetime import timedelta

from dynamik.drift.causality import explain_drift
from dynamik.drift.detection import detect_drift
from dynamik.input import EventMapping
from dynamik.input.csv import read_and_merge_csv_logs
from dynamik.model import Log
from dynamik.output import print_causes, export_causes
from dynamik.utils.logger import LOGGER, Level, setup_logger
from dynamik.utils.pm.concurrency import OverlappingConcurrencyOracle
from dynamik.utils.timer import DEFAULT_TIMER as TIMER


def preprocess(_log: Log) -> Log:
    _log = OverlappingConcurrencyOracle(_log).compute_enablement_timestamps()

    cases = list(set(event.case for event in _log))
    resources = list(set(event.resource for event in _log))
    activities = list(set(event.activity for event in _log))
    roles = list(set(event.attributes["role"] for event in _log))
    work_types = list(set(event.attributes["work_type"] for event in _log))
    classifications = list(set(event.attributes["classification"] for event in _log))
    states = list(set(event.attributes["state"] for event in _log))
    cancellation_reasons = list(set(event.attributes["cancellation_reason"] for event in _log))

    for event in _log:
        event.case = f"case {cases.index(event.case)}"
        event.resource = f"resource {resources.index(event.resource)}" if event.resource is not None else None
        event.activity = f"activity {activities.index(event.activity)}" if event.activity not in ["__SYNTHETIC_START_EVENT__", "__SYNTHETIC_END_EVENT__"] else event.activity
        event.attributes = {
            "role": f"role {roles.index(event.attributes['role'])}" if event.attributes['role'] is not None else None,
            "work_type": f"work type {work_types.index(event.attributes['work_type'])}" if event.attributes['work_type'] is not None else None,
            "classification": f"classification {classifications.index(event.attributes['classification'])}" if event.attributes['classification'] is not None else None,
            "state": f"state {states.index(event.attributes['state'])}" if event.attributes['state'] is not None else None,
            "cancellation_reason": f"reason {cancellation_reasons.index(event.attributes['cancellation_reason'])}" if event.attributes['cancellation_reason'] is not None else None,
        }

    return _log


if __name__ == "__main__":
    with TIMER.profile(__name__):
        setup_logger(Level.NOTICE, destination="output.log", disable_third_party_warnings=True)

        files = (
            ("../data/logs/real/work_orders.csv", "../data/logs/real/work_orders.mapping.json"),
        )

        log = read_and_merge_csv_logs(
            [files[0][0]],
            attribute_mapping=EventMapping.parse(files[0][1]),
            add_artificial_start_end_events=True,
            # preprocessor=lambda _log: OverlappingConcurrencyOracle(_log).compute_enablement_timestamps(),
            preprocessor=preprocess,
        )

        detector = detect_drift(
            log=log,
            timeframe_size=timedelta(days=7),
            overlap_between_models=timedelta(days=0),
            warm_up=timedelta(days=7),
            warnings_to_confirm=3,
        )

        for index, drift in enumerate(detector):
            # causes = explain_drift(drift, first_activity="Sequence A", last_activity="Sequence C")
            causes = explain_drift(drift, first_activity="__SYNTHETIC_START_EVENT__", last_activity="__SYNTHETIC_END_EVENT__")
            tree = export_causes(causes)
            print_causes(causes)
            # with open(f"drift_{index}.json", "w") as file:
            #     json.dump(tree, file, indent=4)

    LOGGER.success("execution took %s", TIMER.elapsed(__name__))