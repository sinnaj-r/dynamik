"""This module contains the definition of the model used for drift detection in the cycle time of a process."""

from __future__ import annotations

import enum
import textwrap
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median, mean, stdev

import numpy as np
from anytree import NodeMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.weightstats import ttost_ind

from dynamik.model import Event

# from dynamik.utils.bayes import probability_difference_under_threshold
from dynamik.utils.logger import LOGGER
from dynamik.utils.model import Pair
from dynamik.utils.pm.batching import discover_batches
from dynamik.utils.pm.processing import ProcessingTimeCanvas
from dynamik.utils.pm.waiting import WaitingTimeCanvas


class DriftCause(NodeMixin):
    """TODO docs"""

    what: str
    how: Pair
    data: Pair
    parent: DriftCause | None
    children: typing.Iterable[DriftCause] | None

    def __init__(
            self: typing.Self,
            what: str,
            how: Pair,
            data: Pair,
            parent: DriftCause | None = None,
            children:  typing.Iterable[DriftCause] | None = None,
    ) -> None:
        super().__init__()
        self.what = what
        self.how = how
        self.data = data
        self.parent = parent
        if children:
            self.children = children

    def __str__(self: typing.Self) -> str:
        return textwrap.dedent(
            f"""\
            {self.what}
                ├── Reference: {self.how.reference}
                └── Running:   {self.how.running}
            """,
        )

    def asdict(self: typing.Self) -> dict:
        """Return a dictionary representation of the object."""
        return {
            "what": self.what,
            "how": self.how.asdict(),
            # "data": self.data.asdict(),
            "causes": [children.asdict() for children in self.children],
        }


class DriftLevel(enum.Enum):
    """The drift level. Can be no drift, drift warning or confirmed drift."""

    NONE = 0
    WARNING = 1
    CONFIRMED = 2


@dataclass
class Drift:
    """The drift, with its level and the data that lead to the detection"""

    level: DriftLevel
    reference_model: Model | None = None
    running_model: Model | None = None
    first_warning: Drift | None = None

    def __post_init__(self: typing.Self) -> None:
        # if the drift has been confirmed, compute the features
        if self.level == DriftLevel.CONFIRMED:
            # compute the batches
            discover_batches(self.reference_model.data)
            discover_batches(self.running_model.data)
            # decompose processing times
            ProcessingTimeCanvas.apply(self.reference_model.data)
            ProcessingTimeCanvas.apply(self.running_model.data)
            # decompose waiting times
            WaitingTimeCanvas.apply(self.reference_model.data)
            WaitingTimeCanvas.apply(self.running_model.data)


NO_DRIFT: Drift = Drift(level=DriftLevel.NONE)


class Model:
    """TODO docs"""

    # The date and time when the model starts
    start: datetime
    # The date and time when the model ends
    end: datetime
    # The collection of events used as the model
    _data: typing.MutableSequence[Event]

    def __init__(self: typing.Self, start: datetime, length: timedelta) -> None:
        self.start = start
        self.end = start + length
        self._data = []

    @property
    def empty(self: typing.Self) -> bool:
        """TODO docs"""
        return len(self.data) == 0

    @property
    def data(self: typing.Self) -> tuple[Event, ...]:
        """An immutable view of the events contained in the model"""
        return tuple(self._data)

    def prune(self: typing.Self) -> None:
        """TODO docs"""
        LOGGER.debug("pruning model")
        # Remove all events that are out of the overlapping region from the running model
        self._data = [event for event in self.data if event.start > self.start and event.end < self.end]

    def add(self: typing.Self, event: Event) -> None:
        """TODO docs"""
        self._data.append(event)

    def statistically_equivalent(
            self: typing.Self,
            other: Model,
            *,
            threshold: timedelta | float = timedelta(minutes=1),
            significance: float = 0.05,
    ) -> bool:
        """TODO docs"""
        if self.empty and other.empty:
            return True
        if self.empty or other.empty:
            return False

        reference_data = [event.cycle_time.total_seconds() for event in self.data]
        running_data = [event.cycle_time.total_seconds() for event in other.data]
        t = threshold

        if isinstance(threshold, float):
            scaler = StandardScaler()
            scaler.fit(np.array(reference_data).reshape(-1, 1))
            reference_data = scaler.transform(np.array(reference_data).reshape(-1, 1)).flatten()
            running_data = scaler.transform(np.array(running_data).reshape(-1, 1)).flatten()
        else:
            t = threshold.total_seconds()

        # only if both models are non-empty, perform the test
        pvalue, _, _ = ttost_ind(reference_data, running_data, -t, t)

        # test, maybe not needed
        # if pvalue > significance:
        #     bayes = probability_difference_under_threshold(reference_data, running_data, t)
        LOGGER.verbose(
            "reference time distribution is mean=%s, median=%s, sd=%s",
            mean(reference_data), median(reference_data), stdev(reference_data),
        )
        LOGGER.verbose(
            "running time distribution is mean=%s, median=%s, sd=%s",
            mean(running_data), median(running_data), stdev(running_data),
        )
        LOGGER.verbose("test(reference != running) p-value: %.4f", pvalue)

        return pvalue <= significance

    def envelopes(self: typing.Self, event: Event) -> bool:
        """TODO docs"""
        return self.start <= event.enabled <= event.end <= self.end

    def update_timeframe(self: typing.Self, start: datetime, length: timedelta) -> None:
        """TODO docs"""
        self.start = start
        self.end = start + length
        # Delete outdated events from the model
        LOGGER.debug("pruning model (timeframe %s - %s)", self.start, self.end)
        self.prune()

    def completed(self: typing.Self, instant: datetime) -> bool:
        """TODO docs"""
        return instant > self.end

    def __repr__(self: typing.Self) -> str:
        return f"""Model(timeframe=({self.start} - {self.end}), events={len(self.data)})"""
