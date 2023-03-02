import logging


def config_file(level: int = logging.DEBUG) -> None:
    """Configure the logger for the application."""
    logging.basicConfig(
        filename='performance_drift.log',
        filemode='w',
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        level=level,
        datefmt='%d/%m/%Y %H:%M:%S',
    )


def config_console(level: int = logging.INFO) -> None:
    """Configure the logger for the application."""
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        level=level,
        datefmt='%d/%m/%Y %H:%M:%S',
    )
