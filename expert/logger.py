"""This module defines the logger used in the entire application."""
import enum
import logging
import math

import coloredlogs
import verboselogs

LOGGER: verboselogs.VerboseLogger = verboselogs.VerboseLogger("expert")
"""The logger instance"""


class Level(enum.Enum):
    """The different configurable logging levels"""

    SPAM=verboselogs.SPAM
    DEBUG=logging.DEBUG
    VERBOSE=verboselogs.VERBOSE
    INFO=logging.INFO
    NOTICE=verboselogs.NOTICE
    WARNING=logging.WARNING
    SUCCESS=verboselogs.SUCCESS
    ERROR=logging.ERROR
    CRITICAL=logging.CRITICAL
    DISABLED=math.inf


def setup_logger(verbosity: Level = Level.INFO) -> None:
    """Configure the log with the provided verbosity level and add colored output"""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    coloredlogs.install(level=verbosity.value, logger=LOGGER)
    LOGGER.setLevel(verbosity.value)



