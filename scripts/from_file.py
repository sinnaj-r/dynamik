from __future__ import annotations

import itertools
from datetime import datetime, timedelta

from dateutil.tz import UTC

import scripts.logger_config as logger
from expert import drift
from expert.input.csv import DEFAULT_APROMORE_CSV_MAPPING, read_csv_log

if __name__ == '__main__':
    start = datetime.now(tz=UTC)

    logger.config_console()
    files = ["../data/Sequence - Normal times 1.csv", "../data/Sequence - Normal times 2.csv"]
    log = (event for event in  itertools.chain(*(read_csv_log(
        file,
        attribute_mapping=DEFAULT_APROMORE_CSV_MAPPING,
        case_prefix=file,
    ) for file in files)))

    drifts = drift.detect_drift(
        log=log,
        timeframe_size=timedelta(minutes=30),
        alpha=0.05,
    )

    end = datetime.now(tz=UTC)

    # if drifts is not None:

    print(end - start)
