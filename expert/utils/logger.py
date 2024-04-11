"""This module defines the logger used in the entire application."""
import enum
import logging
import math
import warnings

import coloredlogs
import verboselogs

LOGGER: verboselogs.VerboseLogger = verboselogs.VerboseLogger("expert")
"""The logger instance"""


class Level(enum.Enum):
    """The different configurable logging levels"""

    SPAM = verboselogs.SPAM
    DEBUG = logging.DEBUG
    VERBOSE = verboselogs.VERBOSE
    INFO = logging.INFO
    NOTICE = verboselogs.NOTICE
    WARNING = logging.WARNING
    SUCCESS = verboselogs.SUCCESS
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    DISABLED = math.inf


def setup_logger(verbosity: Level = Level.INFO, *, destination: str | None = None, disable_third_party_warnings: bool = False) -> None:
    """Configure the log with the provided verbosity level and add colored output"""
    if disable_third_party_warnings:
        warnings.filterwarnings("ignore")

    coloredlogs.install(level=verbosity.value, logger=LOGGER)
    LOGGER.setLevel(verbosity.value)

    pattern = "%(asctime)s %(name)s[%(process)d] %(levelname)8s %(message)s"
    formatter = coloredlogs.ColoredFormatter(pattern)
    for handler in LOGGER.handlers:
        handler.setFormatter(formatter)

    if destination is not None:
        file_handler = logging.FileHandler(destination, mode="w", encoding="utf-8")
        file_handler.setFormatter(coloredlogs.BasicFormatter(pattern))
        LOGGER.addHandler(file_handler)
