import argparse
import json
import logging
import sys
from datetime import timedelta

from expert.drift import detect_drift
from expert.input import EventMapping


def __parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""expert (Explainable Performance Drift) is an algorithm for finding actionable causes for drifts
                       in the performance of a process execution. For this, the cycle time of the process is monitored,
                       and, if a change is detected in the process performance, the algorithm finds the actionable
                       causes for the change.""",
        epilog="expert is licensed under the Apache License, Version 2.0",
    )

    parser.add_argument("log_file", metavar="LOG_FILE", type=str, nargs=1, help="The event log, in CSV or JSON format")
    parser.add_argument("-f", "--format", metavar="FORMAT", choices=["csv", "json"], default="csv",
                        help="specify the event log format")
    parser.add_argument("-m", "--mapping", metavar="MAPPING_FILE", type=str, nargs=1,
                        help="provide a custom mapping file")
    parser.add_argument("-t", "--timeframe", metavar="TIMEFRAME", type=int, nargs=1, default=5,
                        help="provide a timeframe size, in days, used to define the reference and running models")
    parser.add_argument("-a", "--alpha", metavar="ALPHA", type=float, nargs=1, default=0.05,
                        help="specify the confidence for the statistical tests")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="enable verbose output. High verbosity level can drastically decrease expert performance")

    return parser.parse_args()


def __config_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        level=level,
        datefmt='%d/%m/%Y %H:%M:%S',
    )


def __parse_mapping(path: str) -> EventMapping:
    with open(path) as file:
        source = json.load(file)
    return EventMapping(
        start=source["start"],
        end=source["end"],
        resource=source["resource"],
        activity=source["activity"],
        case=source["case"],
    )


__DEBUG = 2
__INFO = 1


def run() -> None:
    args = __parse_arg()

    if args.verbose >= __DEBUG:
        __config_logger(logging.DEBUG)
    elif args.verbose == __INFO:
        __config_logger(logging.INFO)
    else:
        __config_logger(logging.ERROR)

    if args.format == "csv":
        from expert.input.csv import read_csv_log as parser
        if args.mapping is None:
            from expert.input.csv import DEFAULT_CSV_MAPPING as mapping
        else:
            mapping = __parse_mapping(args.mapping)
    elif args.format == "json":
        from expert.input.json import read_json_log as parser
        if args.mapping is None:
            from expert.input.json import DEFAULT_JSON_MAPPING as mapping
        else:
            mapping = __parse_mapping(args.mapping)
    else:
        logging.critical("Log file format is not supported!")
        sys.exit(-1)

    print(f"applying expert drift detector to file {args.log_file}...")

    log = parser(
        args.log_file,
        attribute_mapping=mapping,
    )

    detect_drift(
        log=log,
        timeframe_size=timedelta(days=args.timeframe),
        alpha=args.alpha,
    )

    print("drift detection finished!")
