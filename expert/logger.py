"""This module defines the logger used in the entire application."""

import logging

import coloredlogs
import verboselogs

LOGGER: verboselogs.VerboseLogger = verboselogs.VerboseLogger("expert")
"""The logger instance"""


def setup_logger(verbosity: int = logging.INFO) -> None:
    """Configure the log with the provided verbosity level and add colored output"""
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
    )
    coloredlogs.install(level=verbosity, logger=LOGGER)
    LOGGER.setLevel(verbosity)
