import logging

import coloredlogs


def config_file(level: int = logging.DEBUG) -> None:
    logging.basicConfig(
        filename='execution.log',
        filemode='w',
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        level=level,
        datefmt='%d/%m/%Y %H:%M:%S',
    )


def config_console(level: int = logging.INFO) -> None:
    coloredlogs.install()

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
        level=level,
        datefmt='%d/%m/%Y %H:%M:%S',
    )
