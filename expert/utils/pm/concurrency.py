from __future__ import annotations

import abc
import itertools
import typing
from collections import defaultdict
from dataclasses import dataclass

from expert.model import Event, Log, Trace


@dataclass
class HeuristicsThresholds:
    """Thresholds for the heuristics miner oracle"""

    df: float = 0.9
    l1l: float = 0.9
    l2l: float = 0.9


@dataclass
class OverlappingThresholds:
    """Thresholds for the split miner 2.0 oracle"""

    overlapping_threshold: float = 0.9


class ConcurrencyOracle(abc.ABC):
    """
    A concurrency oracle.

    Allows you to compute the concurrency relations between the events in the log and to set the enablement times for
    every event taking into account these relations.
    """

    log: Log
    concurrency: typing.MutableMapping[str, typing.MutableMapping[str, bool]] = defaultdict(
        lambda: defaultdict(lambda: False)
    )

    def find_enabler(self: typing.Self, trace: Trace, event: Event) -> Event | None:
        """Gets the event enabling the execution of this activity instance within the trace"""
        # Get the list of previous events (events that ended before the current one started and that are not
        # concurrent with it).
        previous = sorted(
            [evt for evt in trace if evt.end <= event.start and not self.concurrency[event.activity][evt.activity]],
            key=lambda evt: evt.end,
        )
        # Return the last
        return previous[-1] if len(previous) > 0 else None

    def compute_enablement_timestamps(self: typing.Self) -> typing.Iterable[Event]:
        """
        Add the enabled time for every event in the log.

        The enabler event is found based on the concurrency relations computed by the HeuristicsMiner oracle.
        Events without an enabler event get their enablement time from their own start time (so, its waiting time is 0).

        Returns
        -------
        * the transformed event log, with the enablement timestamps computed
        """
        # Build traces
        traces = defaultdict(list)
        for event in self.log:
            traces[event.case].append(event)

        for trace in traces.values():
            for event in trace:
                # Find the enabler for the current event
                enabler: Event | None = self.find_enabler(trace, event)
                # Set the enabled timestamp
                if enabler is not None:
                    event.enabled = enabler.end
                else:
                    event.enabled = event.start

        self.log = sorted(self.log, key=lambda evt: (evt.end, evt.start, evt.enabled))

        return self.log


class HeuristicsConcurrencyOracle(ConcurrencyOracle):
    """Concurrency oracle from the heuristics miner."""

    __df_count: typing.Mapping[str, typing.Mapping[str, int]]
    __df_dependency: typing.Mapping[str, typing.Mapping[str, float]]
    __l2l_dependency: typing.Mapping[str, typing.Mapping[str, float]]

    def __init__(
            self: typing.Self,
            log: Log,
            *,
            thresholds: HeuristicsThresholds = HeuristicsThresholds(),
    ) -> None:
        self.log = list(log)

        # Heuristics concurrency
        activities: set[str] = {event.activity for event in self.log}

        # Build dependency matrices
        self.__build_matrices(activities, thresholds)

        # Create concurrency if there is a directly-follows relation in both directions
        self.concurrency = defaultdict(lambda: defaultdict(lambda: False))
        for (activity_1, activity_2) in itertools.combinations(activities, 2):
            if self.__df_count[activity_1][activity_2] > 0 and self.__df_count[activity_2][activity_1] > 0:
                if (self.__l2l_dependency[activity_1][activity_2] < thresholds.l2l and  # 'A' and 'B' are not a length 2 loop
                        abs(self.__df_dependency[activity_1][activity_2] < thresholds.df)):  # The df relations are weak
                    # Concurrency relation AB, add it to A
                    self.concurrency[activity_1][activity_2] = True
                if (self.__l2l_dependency[activity_2][activity_1] < thresholds.l2l and  # 'B' and 'A' are not a length 2 loop
                        abs(self.__df_dependency[activity_2][activity_1]) < thresholds.df):  # The df relations are weak
                    # Concurrency relation AB, add it to A
                    self.concurrency[activity_2][activity_1] = True

    def __build_matrices(
            self: typing.Self,
            activities: typing.Iterable[str],
            thresholds: HeuristicsThresholds,
    ) -> None:
        # Get matrices for:
        # - Directly-follows relations: df_count[A][B] = number of times B following A
        # - Directly-follows dependency values: df_dependency[A][B] = value of certainty that there is a df-relation between A and B
        # - Length-2 loop values: l2l_dependency[A][B] = value of certainty that there is a l2l relation between A and B (A-B-A)

        # Initialize dictionary for directly-follows relations df[A][B] = number of times B following A
        df: typing.MutableMapping[str, typing.MutableMapping[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        # Initialize dictionary for length 2 loops
        l2l: typing.MutableMapping[str, typing.MutableMapping[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))

        # Build traces
        traces: typing.Mapping[str, list[Event]] = defaultdict(list)
        for event in self.log:
            traces[event.case].append(event)

        # Count directly-follows and l2l relations
        for trace in traces.values():
            previous_activity: str | None = None

            # Iterate the events of the trace in pairs: (e1, e2), (e2, e3), (e3, e4)...
            for (event_1, event_2) in itertools.pairwise(trace):
                # Store df relation
                df[event_1.activity][event_2.activity] = df[event_1.activity][event_2.activity] + 1
                # Increase l2l value if there is a length 2 loop (A-B-A)
                if previous_activity and previous_activity == event_2.activity:
                    l2l[previous_activity][event_1.activity] = l2l[previous_activity][event_1.activity] + 1
                # Save previous activity
                previous_activity = event_1.activity
        # Save directly follows counts
        self.__df_count = df

        # Define df and l1l dependency matrices
        df_dependency: typing.MutableMapping[str, typing.MutableMapping[str, float]] = defaultdict(dict)
        l1l_dependency: typing.MutableMapping[str, float] = defaultdict(lambda: 0.0)

        for (activity_1, activity_2) in itertools.combinations_with_replacement(activities, 2):
            if activity_1 == activity_2:
                # Process length 1 loop value
                l1l_dependency[activity_1] = df[activity_1][activity_1] / (df[activity_1][activity_1] + 1.0)
            else:
                ab: float = df[activity_1][activity_2]
                ba: float = df[activity_2][activity_1]
                # Process directly follows dependency value A -> B
                df_dependency[activity_1][activity_2] = (ab - ba) / (ab + ba + 1)
                # Process directly follows dependency value B -> A
                df_dependency[activity_2][activity_1] = (ba - ab) / (ba + ab + 1)
        # Save directly follows dependencies
        self.__df_dependency = df_dependency

        # Define the l2l dependency matrix
        l2l_dependency: typing.MutableMapping[str, typing.MutableMapping[str, float]] = \
            defaultdict(lambda: defaultdict(lambda: 0.0))

        for (activity_1, activity_2) in itertools.combinations(activities, 2):
            if l1l_dependency[activity_1] < thresholds.l1l and l1l_dependency[activity_2] < thresholds.l1l:
                aba: float = l2l[activity_1][activity_2]
                bab: float = l2l[activity_2][activity_1]
                # Process directly follows dependency value A -> B
                l2l_dependency[activity_1][activity_2] = (aba + bab) / (aba + bab + 1)
                # Process directly follows dependency value B -> A
                l2l_dependency[activity_2][activity_1] = (bab + aba) / (bab + aba + 1)
        # Save length-2-loops dependencies
        self.__l2l_dependency = l2l_dependency


class OverlappingConcurrencyOracle(ConcurrencyOracle):
    """Concurrency oracle from the split miner 2.0."""

    __overlaps = defaultdict(lambda: defaultdict(lambda: 0))

    def __init__(
            self: typing.Self,
            log: Log,
            thresholds: OverlappingThresholds = OverlappingThresholds(),
    ) -> None:
        self.log = list(log)

        # build the activity set and the cases map
        activities: set[str] = {event.activity for event in self.log}
        cases: typing.Mapping[str, typing.MutableSequence[Event]] = defaultdict(list)
        for event in self.log:
            cases[event.case].append(event)

        # build overlapping relations
        self.__build_matrices(cases)

        # check overlapping for every pair of activities
        self.concurrency = defaultdict(lambda: defaultdict(lambda: False))
        for (activity_a, activity_b) in itertools.combinations(activities, 2):
            # count instances of activity a
            activity_a_instances = len([
                event for event in self.log if event.activity == activity_a
            ])
            # count instances of activity b
            activity_b_instances = len([
                event for event in self.log if event.activity == activity_b
            ])

            overlap_factor = (0 if (activity_a_instances + activity_b_instances == 0)
                              else 2 * self.__overlaps[activity_a][activity_b] / (activity_a_instances + activity_b_instances))

            if overlap_factor > thresholds.overlapping_threshold:
                # Concurrency relation AB, add it
                self.concurrency[activity_a][activity_b] = True
                self.concurrency[activity_b][activity_a] = True

    def __build_matrices(
            self: typing.Self,
            cases: typing.Mapping[str, Trace],
    ) -> None:
        # Count overlapping relations
        for trace in cases.values():
            for (current, other) in itertools.combinations(trace, 2):
                if (
                        # different activities
                        other.activity != current.activity and
                        (
                                # current starts while other is running
                                other.start < current.start < other.end or
                                # current ends while other is running
                                other.start < current.end < other.end or
                                # current is executed within other's timeframe
                                (other.start <= current.start and current.end <= other.end)
                        )

                ):
                    self.__overlaps[current.activity][other.activity] += 1
                    self.__overlaps[other.activity][current.activity] += 1

                if (
                        # different activities
                        other.activity != current.activity and
                        (
                                # other starts while current is running
                                current.start < other.start < current.end or
                                # other ends while current is running
                                current.start < other.end < current.end or
                                # other is executed within current's timeframe
                                (current.start <= other.start and other.end <= current.end)
                        )

                ):
                    self.__overlaps[other.activity][current.activity] += 1
                    self.__overlaps[current.activity][other.activity] += 1
