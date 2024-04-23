import datetime
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

    def __str__(self: typing.Self) -> str:
        return f"""from {self.reference} to {self.running}"""

    def asdict(self: typing.Self) -> dict:
        """Return a dictionary representation of the object."""
        if hasattr(self.reference, "asdict"):
            return {
                "reference": self.reference.asdict(),
                "running": self.running.asdict(),
            }
        if isinstance(self.reference, typing.Mapping):
            return {
                "reference": {
                    repr(key): value.asdict() for (key, value) in self.reference.items()
                },
                "running": {
                    repr(key): value.asdict() for (key, value) in self.running.items()
                },
            }
        if isinstance(self.reference, typing.Iterable) and hasattr(next(iter(self.reference)), "asdict"):
            return {
                "reference": [item.asdict() for item in self.reference],
                "running": [item.asdict() for item in self.running],
            }
        if isinstance(self.reference, typing.Iterable) and isinstance(next(iter(self.reference)), datetime.timedelta):
            return {
                "reference": [
                    {"days": item.days, "seconds": item.seconds, "microseconds": item.microseconds} for item in self.reference
                ],
                "running": [
                    {"days": item.days, "seconds": item.seconds, "microseconds": item.microseconds} for item in self.running
                ],
            }

        return {
            "reference": self.reference,
            "running": self.running,
        }


@dataclass
class TimeInterval:
    """A collection of time intervals and its total duration."""

    intervals: typing.Iterable[Interval] = ()

    @property
    def duration(self: typing.Self) -> timedelta:
        return sum([interval.end - interval.begin for interval in self.intervals], timedelta())

    def asdict(self: typing.Self) -> dict:
        return {
            "intervals": [
                {
                    "begin": {
                        "year": interval.begin.year,
                        "month": interval.begin.month,
                        "day": interval.begin.day,
                        "hour": interval.begin.hour,
                        "minute": interval.begin.minute,
                        "second": interval.begin.second,
                        "microsecond": interval.begin.microsecond,
                    },
                    "end": {
                        "year": interval.end.year,
                        "month": interval.end.month,
                        "day": interval.end.day,
                        "hour": interval.end.hour,
                        "minute": interval.end.minute,
                        "second": interval.end.second,
                        "microsecond": interval.end.microsecond,
                    },
                } for interval in self.intervals
            ],
            "duration": {
                "days": self.duration.days,
                "seconds": self.duration.seconds,
                "microseconds": self.duration.microseconds,
            },
        }


class TestResult(typing.NamedTuple):
    """The result of a statistical test. This class is just a mock of the one in scipy.stats._stats_py.SignificanceResult"""

    statistic: float
    pvalue: float


@dataclass
class DistributionDescription:
    """The description of a distribution. This class is just a mock of the one in scipy.stats._stats_py.DescribeResult"""

    nobs: int
    minmax: tuple[float, float]
    mean: float
    variance: float
    skewness: float
    kurtosis: float

    def __init__(self: typing.Self, data: tuple) -> None:
        self.nobs = data.nobs
        self.minmax = data.minmax
        self.mean = data.mean
        self.variance = data.variance
        self.skewness = data.skewness
        self.kurtosis = data.kurtosis

    def asdict(self: typing.Self) -> dict:
        return {
            "nobs": self.nobs,
            "minmax": self.minmax,
            "mean": self.mean,
            "variance": self.variance,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }