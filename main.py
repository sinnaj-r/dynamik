from __future__ import annotations

import logging
from datetime import timedelta

import pandas

import drift
from fields import DEFAULT_EVENT_FIELDS
from input import read_csv_log


def run(
        log_path_1: str,
        log_path_2: str,
        timeframe_size: timedelta,
        event_fields=DEFAULT_EVENT_FIELDS
):
    log_1 = read_csv_log(log_path_1, event_fields=event_fields)
    log_1[event_fields.CASE] = 'log1_' + log_1[event_fields.CASE].astype(str)
    log_2 = read_csv_log(log_path_2, event_fields=event_fields)
    log_2[event_fields.CASE] = 'log2_' + log_2[event_fields.CASE].astype(str)

    log = pandas.concat([log_1, log_2])

    print('finding drifts')

    drifts = drift.detect_drift(
        log=log,
        timeframe_size=timeframe_size
    )

    print(f'{len(drifts)} drifts detected!')
    # print('\n'.join([f'\t{event}' for event in drifts]))


if __name__ == '__main__':
    logging.basicConfig(
        filename='debug.log',
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        level=logging.CRITICAL,
        datefmt='%d/%m/%Y %H:%M:%S'
    )

    log1 = 'data/Sequence - Normal times 1.csv'
    log2 = 'data/Sequence - Normal times 2.csv'

    run(log_path_1=log1, log_path_2=log2, timeframe_size=timedelta(days=5))
