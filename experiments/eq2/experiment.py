from __future__ import annotations

from datetime import timedelta

from dynamik.drift import detect_drift, explain_drift
from dynamik.drift.model import DriftLevel
from dynamik.input import EventMapping
from dynamik.input.csv import read_and_merge_csv_logs
from dynamik.output import print_causes
from dynamik.utils.logger import Level, setup_logger

if __name__ == "__main__":
    setup_logger(Level.NOTICE, destination="output.log", disable_third_party_warnings=True)

    mapping = EventMapping.parse("./mappings/prosimos.mapping.json")

    logs = (
            (
                # 14 days, 3, 30 min
                "./logs/scenario10.log.csv",
                "./logs/scenario11.log.csv",
            ),
            (
                # 14 days, 1, 5 min
                "./logs/scenario20.log.csv",
                "./logs/scenario21.log.csv",
            ),
            (
                # 14 days, 3, 30 min
                "./logs/scenario30.log.csv",
                "./logs/scenario31.log.csv",
            ),
            (
                # 14 days, 3, 30 min
                "./logs/scenario40.log.csv",
                "./logs/scenario41.log.csv",
            ),
            (
                # 7 days, 1, 5 min
                "./logs/scenario50.log.csv",
                "./logs/scenario51.log.csv",
            ),
    )

    for _logs in logs:
        print(_logs)

        log = read_and_merge_csv_logs(_logs, attribute_mapping=mapping)

        detector = detect_drift(
            log=log,
            timeframe_size=timedelta(days=14),
            warm_up=timedelta(),
            warnings_to_confirm=3,
            threshold=timedelta(minutes=30),
        )

        for drift in detector:
            if drift.level == DriftLevel.CONFIRMED:
                print(drift)
                causes = explain_drift(
                    drift,
                    first_activity='__SYNTHETIC_START_EVENT__',
                    last_activity='__SYNTHETIC_END_EVENT__',
                )
                print_causes(causes)
