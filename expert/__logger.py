import logging

import coloredlogs
import verboselogs

LOGGER: verboselogs.VerboseLogger = verboselogs.VerboseLogger("expert")

def setup_logger(verbosity: int = logging.INFO) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
    )
    coloredlogs.install(level=verbosity, logger=LOGGER)
    LOGGER.setLevel(verbosity)
