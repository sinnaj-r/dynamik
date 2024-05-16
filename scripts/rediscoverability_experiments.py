from __future__ import annotations

import json
import os
from datetime import timedelta

from expert.drift import detect_drift, explain_drift
from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.output import print_causes, export_causes
from expert.utils.logger import LOGGER, Level, setup_logger
from expert.utils.timer import DEFAULT_TIMER as TIMER


if __name__ == "__main__":
    base_scenario_log = "../data/logs/base.log.csv"
    mapping = "../data/mappings/prosimos.mapping.json"
    alternative_scenarios = (
        # x "batching",
        # x "contention1",
        # x "contention2",
        # "prioritization",
        # x "unavailability1",
        # "unavailability2",
    )

    for alternative_scenario in alternative_scenarios:
        # create directory for results
        os.makedirs(f"../data/results/{alternative_scenario}/causes/", exist_ok=True)
        setup_logger(Level.NOTICE, destination=f"../data/results/{alternative_scenario}/output.log", disable_third_party_warnings=True)

        log = read_and_merge_csv_logs(
            (
                base_scenario_log,
                f"../data/logs/{alternative_scenario}.log.csv",
            ),
            attribute_mapping=EventMapping.parse(mapping),
        )

        TIMER.start(__name__)

        detector = detect_drift(
            log=log,
            timeframe_size=timedelta(days=7),
            overlap_between_models=timedelta(days=0),
            warm_up=timedelta(days=1),
            warnings_to_confirm=3,
        )

        for index, drift in enumerate(detector):
            causes = explain_drift(drift, first_activity="A", last_activity="E")
            tree = export_causes(causes)
            print_causes(causes)
            with open(f"../data/results/{alternative_scenario}/causes/drift-{index}.causes.json", "w") as file:
                json.dump(tree, file, indent=4)

        TIMER.end(__name__)
        LOGGER.success(f"{alternative_scenario} scenario execution took %s", TIMER.elapsed(__name__))
        TIMER.reset(__name__)
