from __future__ import annotations

from datetime import datetime, timedelta

from dynamik.drift import detect_drift
from dynamik.input import EventMapping
from dynamik.input.csv import read_and_merge_csv_logs
from dynamik.utils.logger import Level, setup_logger


def _get_drift_date(name: str) -> datetime:
    with open(f"./stats/{name}.stat.csv") as file:
        lines = file.readlines()
        return datetime.fromisoformat(lines[0].split(",")[1].strip())


if __name__ == "__main__":
    setup_logger(Level.NOTICE, destination="output.log", disable_third_party_warnings=True)

    params = {
        "timeframes": (timedelta(days=7), timedelta(days=14)),
        "warnings": (1, 3),
        "thresholds": (timedelta(minutes=1), timedelta(minutes=10), timedelta(minutes=30), timedelta(hours=1)),
    }

    mapping = EventMapping.parse("./mappings/prosimos.mapping.json")

    logs = (
        (
            (
                "./logs/scenario10.log.csv",
                "./logs/scenario11.log.csv",
                "./logs/scenario12.log.csv",
            ),
            (
                _get_drift_date("scenario11"),
                _get_drift_date("scenario12"),
            ),
            timedelta(minutes=1),
        ),
        (
            (
                "./logs/scenario20.log.csv",
                "./logs/scenario21.log.csv",
            ),
            (
                _get_drift_date("scenario21"),
            ),
            timedelta(minutes=5),
        ),
        (
            (
                "./logs/scenario30.log.csv",
                "./logs/scenario31.log.csv",
                "./logs/scenario32.log.csv",
                "./logs/scenario33.log.csv",
                "./logs/scenario34.log.csv",
            ),
            (
                _get_drift_date("scenario31"),
                _get_drift_date("scenario32"),
                _get_drift_date("scenario33"),
                _get_drift_date("scenario34"),
            ),
            timedelta(minutes=10),
        ),
        (
            (
                "./logs/scenario40.log.csv",
                "./logs/scenario41.log.csv",
            ),
            (
                _get_drift_date("scenario41"),
            ),
            timedelta(minutes=15),
        ),
        (
            (
                "./logs/scenario50.log.csv",
                "./logs/scenario51.log.csv",
                "./logs/scenario52.log.csv",
                "./logs/scenario53.log.csv",
            ),
            (
                _get_drift_date("scenario51"),
                _get_drift_date("scenario52"),
                _get_drift_date("scenario53"),
            ),
            timedelta(minutes=20),
        ),
        (
            (
                "./logs/scenario60.log.csv",
                "./logs/scenario61.log.csv",
                "./logs/scenario62.log.csv",
            ),
            (
                _get_drift_date("scenario61"),
                _get_drift_date("scenario62"),
            ),
            timedelta(minutes=30),
        ),
        (
            (
                "./logs/scenario70.log.csv",
                "./logs/scenario71.log.csv",
            ),
            (
                _get_drift_date("scenario71"),
            ),
            timedelta(minutes=45),
        ),
        (
            (
                "./logs/scenario80.log.csv",
                "./logs/scenario81.log.csv",
            ),
            (
                _get_drift_date("scenario81"),
            ),
            timedelta(minutes=60),
        ),
        (
            (
                "./logs/scenario90.log.csv",
                "./logs/scenario91.log.csv",
                "./logs/scenario92.log.csv",
                "./logs/scenario93.log.csv",
                "./logs/scenario94.log.csv",
                "./logs/scenario95.log.csv",
            ),
            (
                _get_drift_date("scenario91"),
                _get_drift_date("scenario92"),
                _get_drift_date("scenario93"),
                _get_drift_date("scenario94"),
                _get_drift_date("scenario95"),
            ),
            timedelta(minutes=90),
        ),
        (
            (
                "./logs/scenario100.log.csv",
                "./logs/scenario101.log.csv",
            ),
            (
                _get_drift_date("scenario101"),
            ),
            timedelta(minutes=120),
        ),
    )

    accuracy = {}
    delay = {}

    for size in params["timeframes"]:
        accuracy[size] = {}
        delay[size] = {}
        for warnings in params["warnings"]:
            accuracy[size][warnings] = {}
            delay[size][warnings] = {}
            for threshold in params["thresholds"]:
                accuracy[size][warnings][threshold] = {}
                delay[size][warnings][threshold] = []

                tp, fp, fn = 0, 0, 0

                for (_logs, _drifts, _diff) in logs:
                    print(_logs, size, warnings, threshold)

                    log = read_and_merge_csv_logs(_logs, attribute_mapping=mapping)

                    detector = detect_drift(
                        log=log,
                        timeframe_size=size,
                        warm_up=timedelta(),
                        warnings_to_confirm=warnings,
                        threshold=threshold,
                        significance=0.05,
                    )

                    results = list(detector)

                    # if drifts have to be detected
                    if threshold <= _diff:
                        pairings = []
                        for _drift in _drifts:
                            changes_in_region = [
                                change for change in results if
                                _drift - (size * warnings * 2) < change.running_model.end < _drift + (size * warnings * 2)
                            ]

                            pairings.append((_drift, changes_in_region[0]) if len(changes_in_region) > 0 else (_drift, None))
                            if len(changes_in_region) > 0:
                                delay[size][warnings][threshold].append(abs(_drift - changes_in_region[0].running_model.end))

                        _tp = len([real for real, detected in pairings if detected is not None])
                        _fn = len([real for real, detected in pairings if detected is None])
                        _fp = len(results) - _tp

                        tp += _tp
                        fp += _fp
                        fn += _fn

                # accuracy
                accuracy[size][warnings][threshold] = (2*tp)/(2*tp+fp+fn)
                # delay is the minimum distance from a drift detected in the change region to the real drift point
                delay[size][warnings][threshold] = sum(delay[size][warnings][threshold], timedelta())/len(delay[size][warnings][threshold])

    # print
    for size in params["timeframes"]:
        for warnings in params["warnings"]:
            for threshold in params["thresholds"]:
                print(f"({size.days}, {warnings}, {threshold.seconds / 60}) & {accuracy[size][warnings][threshold]} & {delay[size][warnings][threshold]}")  # noqa: E501
