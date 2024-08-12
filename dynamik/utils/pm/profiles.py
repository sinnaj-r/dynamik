from __future__ import annotations

import abc
import functools
import itertools
import typing
from collections import defaultdict
from datetime import timedelta

import scipy
from intervaltree import Interval
from statsmodels.stats.weightstats import ttost_ind

from dynamik.model import Activity, Log, Resource
from dynamik.utils.pm.calendars import Calendar


class Profile(abc.ABC):
    """TODO docs"""

    @abc.abstractmethod
    def statistically_equals(self: typing.Self, other: Profile, significance: float = 0.05) -> bool:
        """TODO docs"""


class ActivityProfile(Profile):
    """"TODO docs"""

    activities: typing.Iterable[Activity]
    # requirements
    activity_frequency: typing.MutableMapping[Activity, int]
    # demand
    demand: typing.MutableMapping[Activity, float]
    # behaviour
    arrival_distribution: typing.MutableMapping[Activity, Calendar]
    # complexity
    complexity_deviation: typing.MutableMapping[Activity, typing.Iterable[float]]
    # interactions
    co_occurrence_index: typing.MutableMapping[tuple[Activity, Activity], int]

    def __init__(
            self: typing.Self,
            activities: typing.Iterable[Activity],
            activity_frequency: typing.MutableMapping[Activity, int],
            demand: typing.MutableMapping[Activity, float],
            arrival_distribution: typing.MutableMapping[Activity, Calendar],
            complexity_deviation: typing.MutableMapping[Activity, typing.Iterable[float]],
            co_occurrence_index: typing.MutableMapping[tuple[Activity, Activity], int],
    ) -> None:
        self.activities = activities
        self.activity_frequency = activity_frequency
        self.demand = demand
        self.arrival_distribution = arrival_distribution
        self.complexity_deviation = complexity_deviation
        self.co_occurrence_index = co_occurrence_index

    def __str__(self: typing.Self) -> str:
        return str(self.asdict())

    def statistically_equals(self: typing.Self, other: ActivityProfile, significance: float = 0.05) -> bool:
        """TODO docs"""
        # compute the sets of all activities and resources present in any of the profiles
        activities = set(itertools.chain(self.activities, other.activities))

        instance_count_drift = {}
        demand_drift = {}
        arrival_distribution_drift = {}
        complexity_deviation_drift = {}
        co_occurrence_index_drift = {}

        for activity in activities:
            # check activity count
            instance_count_drift[activity] = scipy.stats.poisson_means_test(
                # check the count of instances for an activity in self
                self.activity_frequency[activity],
                # wrt. the total instances in self, vs
                sum(self.activity_frequency.values()),
                # the count of instances for an activity in other
                other.activity_frequency[activity],
                # wrt. the total instances in other
                sum(other.activity_frequency.values()),
            ).pvalue < significance

            # check workload demand
            demand_drift[activity] = scipy.stats.poisson_means_test(
                # check the percentage of time executing activity in self, vs
                int(self.demand[activity] * 100),
                100,
                # the percentage of time executing activity in other
                int(other.demand[activity] * 100),
                100,
            ).pvalue < significance

            # check arrival distribution
            arrival_distribution_drift[activity] = self.arrival_distribution[activity].statistically_equals(
                other.arrival_distribution[activity],
            )

            # check complexity deviation
            pvalue, _, _ = ttost_ind(self.complexity_deviation[activity], other.complexity_deviation[activity], 0, 0)
            complexity_deviation_drift[activity] = pvalue > significance

            # check co-occurrence index
            for activity2 in activities:
                # check only activities that are present in both models
                # (the addition of new activities is already contemplated in the activity frequency aspect)
                if (
                        activity in self.activities and
                        activity in other.activities and
                        activity2 in self.activities and
                        activity2 in other.activities
                ):
                    co_occurrence_index_drift[(activity, activity2)] = scipy.stats.poisson_means_test(
                        # check the count of co-occurrences in self
                        self.co_occurrence_index[(activity, activity2)],
                        # wrt. the sum of cases where any of the activities is executed in self
                        # (A1 + A2 - A1A2, subtract the co-occurrences to not count them twice), vs
                        self.co_occurrence_index[(activity, activity)]
                        + self.co_occurrence_index[(activity2, activity2)]
                        - self.co_occurrence_index[(activity, activity2)],
                        # the count of co-occurrences in other
                        other.co_occurrence_index[(activity, activity2)],
                        # wrt. the sum of cases where any of the activities is executed in other
                        # (A1 + A2 - A1A2, subtract the co-occurrences to not count them twice)
                        other.co_occurrence_index[(activity, activity)]
                        + other.co_occurrence_index[(activity2, activity2)]
                        - other.co_occurrence_index[(activity, activity2)],
                    ).pvalue < significance

        return any([
            any(instance_count_drift.values()),
            any(demand_drift.values()),
            any(arrival_distribution_drift.values()),
            any(complexity_deviation_drift.values()),
            any(co_occurrence_index_drift.values()),
        ])

    def asdict(self: typing.Self) -> dict:
        return {
            "activities": list(self.activities),
            "activity_frequency": self.activity_frequency,
            "demand": self.demand,
            "arrival_distribution": {
                key: value.asdict() for (key, value) in self.arrival_distribution.items()
            },
            "complexity_deviation": self.complexity_deviation,
            "co_occurrence_index": [
                {"activities": [act1, act2], "count": count} for (act1, act2), count in self.co_occurrence_index.items() if count > 0
            ],
        }

    @staticmethod
    @functools.lru_cache
    def discover(log: Log) -> ActivityProfile:
        """TODO docs"""
        activities = {event.activity for event in log if not event.activity.startswith('__')}
        activity_profile = ActivityProfile(
            activities=activities,
            activity_frequency=defaultdict(lambda: 0),
            demand=defaultdict(lambda: 0.0),
            arrival_distribution=defaultdict(Calendar),
            complexity_deviation=defaultdict(list),
            co_occurrence_index=defaultdict(lambda: 0),
        )

        for activity in activities:
            activity_instances = [event for event in log if event.activity == activity]

            # compute activity frequency for each activity
            activity_profile.activity_frequency[activity] = len(activity_instances)

            # compute the workforce demand
            required_work = sum([event.processing_time.effective.duration for event in activity_instances], timedelta())
            total_work = sum([event.processing_time.effective.duration for event in log], timedelta())
            activity_profile.demand[activity] = required_work / total_work

            # compute the arrival distribution
            activity_profile.arrival_distribution[activity] = Calendar.discover(activity_instances, lambda event: [event.enabled])

            # compute the complexity deviation
            mean_execution_time = sum([event.processing_time.effective.duration for event in log], timedelta()) / len(list(log))
            # compute the deviation for each activity instance vs the average (the deviation is the factor of event time vs
            # average time, i.e., <1.0 means over performance, >1.0 means under performance)
            activity_profile.complexity_deviation[activity] = [event.processing_time.effective.duration / mean_execution_time for event in activity_instances]

            # compute the co-occurrence index
            # get the set of own cases
            own_cases = {event.case for event in activity_instances}
            # build a map with the activities executed in each of the own cases
            activities_per_case = {case: {event.activity for event in log if event.case == case} for case in own_cases}

            for co_occurrence in activities:
                common_cases = {case for case in activities_per_case if co_occurrence in activities_per_case[case]}
                activity_profile.co_occurrence_index[(activity, co_occurrence)] = len(common_cases)

        return activity_profile


class ResourceProfile(Profile):
    """TODO docs"""

    resources: typing.Iterable[Resource]
    # skills
    instance_count: typing.MutableMapping[Resource, int]
    # utilization
    utilization_index: typing.MutableMapping[Resource, float]
    # preferences
    effort_distribution: typing.MutableMapping[Resource, Calendar]
    # productivity
    performance_deviation: typing.MutableMapping[Resource, typing.MutableMapping[Activity, typing.Iterable[float]]]
    # collaboration
    collaboration_index: typing.MutableMapping[tuple[Resource, Resource], int]

    def __init__(
            self: typing.Self,
            resources: typing.Iterable[Resource],
            instance_count: typing.MutableMapping[Resource, int],
            utilization_index: typing.MutableMapping[Resource, float],
            effort_distribution: typing.MutableMapping[Resource, Calendar],
            performance_deviation: typing.MutableMapping[Resource, typing.MutableMapping[Activity, typing.Iterable[float]]],
            collaboration_index: typing.MutableMapping[tuple[Resource, Resource], int],
    ) -> None:
        self.resources = resources
        self.instance_count = instance_count
        self.utilization_index = utilization_index
        self.effort_distribution = effort_distribution
        self.performance_deviation = performance_deviation
        self.collaboration_index = collaboration_index

    def __str__(self: typing.Self) -> str:
        return str(self.asdict())

    def statistically_equals(self: typing.Self, other: ResourceProfile, significance: float = 0.05) -> bool:
        """TODO docs"""
        # compute the sets of all resources present in any of the profiles
        all_resources = set(itertools.chain(self.resources, other.resources))

        instance_count_drift = {}
        utilization_index_drift = {}
        effort_distribution_drift = {}
        performance_deviation_drift = {}
        collaboration_index_drift = {}

        for resource in all_resources:
            # check instance count
            instance_count_drift[resource] = scipy.stats.poisson_means_test(
                # check the count of instances for resource in self
                self.instance_count[resource],
                # wrt. the total instances executed in self, vs
                sum(self.instance_count.values()),
                # the count of instances for resource in other
                other.instance_count[resource],
                # wrt. the total instances in other
                sum(other.instance_count.values()),
            ).pvalue < significance

            # check utilization index
            utilization_index_drift[resource] = scipy.stats.poisson_means_test(
                # check the percentage of time the resource is busy in self, vs
                int(self.utilization_index[resource] * 100),
                100,
                # the percentage of time the resource is busy in other
                int(other.utilization_index[resource] * 100),
                100,
            ).pvalue < significance

            # check effort distribution
            effort_distribution_drift[resource] = self.effort_distribution[resource].statistically_equals(
                other.effort_distribution[resource],
            )

            # check performance deviation
            pvalues = []
            for activity in set(self.performance_deviation[resource]).union(other.performance_deviation[resource]):
                if activity not in self.performance_deviation[resource] or activity not in other.performance_deviation[resource]:
                    pvalue = 1
                else:
                    pvalue, _, _ = ttost_ind(self.performance_deviation[resource][activity], other.performance_deviation[resource][activity], 0, 0)
                pvalues.append(pvalue)
            performance_deviation_drift[resource] = any(pvalue > significance for pvalue in pvalues)

            # check collaboration index
            for resource2 in all_resources:
                # check only common resources that are present in both models
                # (the addition of new resources is already contemplated in the resource frequency aspect)
                if (
                        resource in self.resources and
                        resource in other.resources and
                        resource2 in self.resources and
                        resource2 in other.resources
                ):
                    collaboration_index_drift[(resource, resource2)] = scipy.stats.poisson_means_test(
                        # check the count of collaborations in self
                        self.collaboration_index[(resource, resource2)],
                        # wrt. the sum of cases executed by resource 1 or resource 2 in self, vs
                        self.collaboration_index[(resource, resource)]
                        + self.collaboration_index[(resource2, resource2)]
                        - self.collaboration_index[(resource, resource2)],
                        # the count of instances for an activity in other
                        other.collaboration_index[(resource, resource2)],
                        # wrt. the total instances executed by resource 1 or resource 2 in other
                        other.collaboration_index[(resource, resource)]
                        + other.collaboration_index[(resource2, resource2)]
                        - other.collaboration_index[(resource, resource2)],
                    ).pvalue < significance

        return any([
            any(instance_count_drift.values()),
            any(utilization_index_drift.values()),
            any(effort_distribution_drift.values()),
            any(performance_deviation_drift.values()),
            any(collaboration_index_drift.values()),
        ])

    def asdict(self: typing.Self) -> dict:
        return {
            'resources': list(self.resources),
            'instance_count': self.instance_count,
            'utilization_index': self.utilization_index,
            'effort_distribution': {
                key: value.asdict() for (key, value) in self.effort_distribution.items()
            },
            'performance_deviation': self.performance_deviation,
            'collaboration_index': [
                {"resources": [res1, res2], "count": count} for (res1, res2), count in self.collaboration_index.items() if count > 0
            ],
        }

    @staticmethod
    @functools.lru_cache
    def discover(log: Log) -> ResourceProfile:
        """TODO docs"""
        activities = {event.activity for event in log if not event.activity.startswith('__')}
        resources = {event.resource for event in log}
        resource_profile = ResourceProfile(
            resources=resources,
            instance_count=defaultdict(lambda: 0),
            utilization_index=defaultdict(lambda: 0.0),
            effort_distribution=defaultdict(Calendar),
            performance_deviation=defaultdict(lambda: defaultdict(list)),
            collaboration_index=defaultdict(lambda: 0),
        )

        for resource in resources:
            events_by_resource = [event for event in log if event.resource == resource]

            # compute activity instance count for each activity executed by the resource
            resource_profile.instance_count[resource] = len(events_by_resource)

            # compute the utilization index
            worked_time = sum([event.processing_time.effective.duration for event in events_by_resource], timedelta())
            available_time = sum([
                interval.end - interval.begin for interval in Calendar.discover(events_by_resource).apply(
                    Interval(begin=min(event.start for event in log), end=max(event.end for event in log)),
                )
            ], timedelta())
            resource_profile.utilization_index[resource] = worked_time/available_time

            # compute the effort distribution
            resource_profile.effort_distribution[resource] = Calendar.discover(events_by_resource)

            # compute the performance deviation
            deviations = {}
            for activity in activities:
                self_events = [event for event in events_by_resource if event.activity == activity]
                all_events = [event for event in log if event.activity == activity]
                mean_execution_time = sum([event.processing_time.effective.duration for event in all_events], timedelta()) / len(all_events)
                # check deviations only when any event is present for the resource
                if len(self_events) > 0 and len(all_events) > 0:
                    # compute the deviation for each event vs the average (the deviation is the factor of event time vs
                    # average time, i.e., <1.0 means over performance, >1.0 means under performance)
                    deviations[activity] = [event.processing_time.effective.duration / mean_execution_time for event in self_events]

            resource_profile.performance_deviation[resource] = deviations

            # compute the collaboration index
            # get the set of own cases
            own_cases = {event.case for event in events_by_resource}
            # build a map with the resources involved in each of the own cases
            resources_per_case = {case: {event.resource for event in log if event.case == case} for case in own_cases}

            for collaborator in resources:
                cases_collaborated = {case for case in resources_per_case if collaborator in resources_per_case[case]}
                # the collaboration index is the ratio between the shared cases and the own cases
                resource_profile.collaboration_index[(resource, collaborator)] = len(cases_collaborated)

        return resource_profile
