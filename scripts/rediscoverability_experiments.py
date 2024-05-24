from __future__ import annotations

import itertools
import multiprocessing
from datetime import timedelta

from dynamik.input import EventMapping
from runner import Runner


if __name__ == "__main__":
    params = {
        "timeframes": (timedelta(days=7), timedelta(days=14),),
        "warmups": (timedelta(days=1), timedelta(days=7),),
        "warnings": (3, 1),
        "thresholds": (0.1, 0.2, 0.5, timedelta(minutes=10), timedelta(minutes=30), timedelta(hours=1)),
        "overlaps": (0.0, 0.5),
    }

    base_scenario = "../data/logs/base.log.csv"
    mapping = EventMapping.parse("../data/mappings/prosimos.mapping.json")
    alternative_scenarios = (
        "batching",
        "contention1",
        "contention2",
        "unavailability1",
        # # "prioritization",
        # # "unavailability2",
    )

    with multiprocessing.Pool(6) as pool:
        for alternative_scenario in alternative_scenarios:
            runner = Runner(base_scenario, alternative_scenario, mapping)
            data_outputs = pool.map(
                runner,
                itertools.product(
                    params["timeframes"],
                    params["warmups"],
                    params["warnings"],
                    params["thresholds"],
                    params["overlaps"],
                )
            )
