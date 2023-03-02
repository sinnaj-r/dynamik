from __future__ import annotations

import itertools
from datetime import timedelta

import scripts.logger_config as logger
from epd import drift
from epd.input.csv import DEFAULT_APROMORE_CSV_MAPPING, read_csv_log

if __name__ == '__main__':
    logger.config_console()
    files = ["../data/Sequence - Normal times 1.csv", "../data/Sequence - Normal times 2.csv"]
    log = itertools.chain(*(read_csv_log(
        file,
        attribute_mapping=DEFAULT_APROMORE_CSV_MAPPING,
        filter_events_without_resources=False,
    ) for file in files))

    drifts = drift.detect_drift(
        log=log,
        timeframe_size=timedelta(days=5),
        alpha=0.005,
    )

    print(f'{len(list(drifts))} drifts detected!')
