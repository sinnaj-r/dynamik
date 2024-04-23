"""This module contains the logic for plotting change causes"""

from anytree import RenderTree

from expert.drift.model import DriftCause
from expert.utils.logger import LOGGER
from expert.utils.timer import profile


@profile()
def print_causes(drift_causes: DriftCause) -> None:
    """Pretty-print the drift causes"""
    tree = str(RenderTree(drift_causes).by_attr(lambda node: str(node)))
    for line in tree.splitlines():
        LOGGER.notice(line)


@profile()
def export_causes(drift_causes: DriftCause) -> dict:
    """Export the current causes tree to a dict"""
    # transform the tree to a dict and return it
    return drift_causes.asdict()
