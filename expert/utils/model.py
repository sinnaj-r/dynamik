import typing
from dataclasses import dataclass
from datetime import timedelta

from intervaltree import Interval

_T = typing.TypeVar("_T")


@dataclass
class Pair(typing.Generic[_T]):
    """A pair of measurement values, used as reference and running values for drift detection"""

    reference: _T | None
    """The reference values"""
    running: _T | None
    """The running values"""


@dataclass
class TimeInterval:
    """A collection of time intervals and its total duration."""

    intervals: typing.Iterable[Interval] = ()
    duration: timedelta = timedelta()
