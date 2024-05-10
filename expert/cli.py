"""This module contains the logic used for launching expert from the command line."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

from rich_argparse import RichHelpFormatter

from expert.drift import detect_drift, explain_drift
from expert.input import EventMapping
from expert.input.csv import DEFAULT_CSV_MAPPING as MAPPING
from expert.input.csv import read_and_merge_csv_logs as parse
from expert.output import export_causes, print_causes
from expert.utils.logger import LOGGER, Level, setup_logger
from expert.utils.pm.concurrency import OverlappingConcurrencyOracle
from expert.utils.timer import DEFAULT_TIMER as TIMER

LEVELS = [Level.NOTICE, Level.INFO, Level.VERBOSE, Level.DEBUG, Level.SPAM]


def __parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""expert (Explainable Performance Drift) is an algorithm for finding actionable causes for drifts
                       in the performance of a process execution. For this, the cycle time of the process is monitored,
                       and, if a change is detected in the process performance, the algorithm finds the actionable
                       causes for the change.""",
        epilog="expert is licensed under the Apache License, Version 2.0",
        formatter_class=RichHelpFormatter,
    )

    parser.add_argument("log_files", metavar="LOG_FILES", type=str, nargs="+",
                        help="The event logs, in CSV format")
    parser.add_argument("-o", "--output", metavar="OUTPUT", type=str, default="./",
                        help="The destination for the output files")
    parser.add_argument("-m", "--mapping", metavar="MAPPING_FILE", type=str,
                        help="provide a custom mapping file")
    parser.add_argument("-t", "--timeframe", metavar="TIMEFRAME", type=int, default=5,
                        help="provide a timeframe size, in days, used to define the reference and running models")
    parser.add_argument("-u", "--warmup", metavar="WARMUP", type=int, default=5,
                        help="provide the number of days used as a warm-up")
    parser.add_argument("-p", "--overlap", metavar="OVERLAP", type=int, default=0,
                        help="provide the overlap between reference and running models, in days")
    parser.add_argument("-w", "--warnings", metavar="WARNINGS", type=int, default=3,
                        help="provide a number of warnings to wait after confirming a drift")
    parser.add_argument("-e", "--explain", action="store_true", default=False,
                        help="explain the found drifts")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="enable verbose output. WARNING: high verbosity levels can drastically decrease performance!")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="disable all output")

    return parser.parse_args()


def run() -> None:
    args = __parse_arg()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.quiet:
        setup_logger(
            Level.DISABLED,
            disable_third_party_warnings=True,
        )
    else:
        setup_logger(
            LEVELS[args.verbose],
            destination=os.path.join(args.output, "execution.log"),
            disable_third_party_warnings=True,
        )

    mapping = EventMapping.parse(args.mapping) if args.mapping is not None else MAPPING

    LOGGER.notice("applying expert drift detector to files %s", ", ".join(args.log_files))
    LOGGER.notice("results will be saved to %s", args.output)

    with TIMER.profile(__name__):
        log = parse(
            args.log_files,
            attribute_mapping=mapping,
            add_artificial_start_end_events=True,
            preprocessor=lambda _log: _log if mapping.enablement is not None else OverlappingConcurrencyOracle(_log).compute_enablement_timestamps(),
        )

        detector = detect_drift(
            log=log,
            timeframe_size=timedelta(days=args.timeframe),
            warm_up=timedelta(days=args.warmup),
            warnings_to_confirm=args.warnings,
            overlap_between_models=timedelta(days=args.overlap),
        )

        for index, drift in enumerate(detector):
            if args.explain:
                causes = explain_drift(drift, first_activity="__SYNTHETIC_START_EVENT__", last_activity="__SYNTHETIC_END_EVENT__")

                with open(os.path.join(args.output, f"drift_{index}.json"), "w") as file:
                    json.dump(export_causes(causes), file, indent=4)

                LOGGER.notice("causes:")
                print_causes(causes)

    LOGGER.success("execution took %s", TIMER.elapsed(__name__))
