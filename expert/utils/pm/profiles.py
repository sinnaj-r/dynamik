from __future__ import annotations

import abc
import itertools
import typing
from collections import defaultdict
from datetime import timedelta

import scipy
from intervaltree import Interval

from expert.process_model import Activity, Log, Resource
from expert.utils.pm.calendars import Calendar
from expert.utils.timer import profile


class Profile(abc.ABC):
    """TODO docs"""

    @abc.abstractmethod
    def statistically_equals(self: typing.Self, other: Profile, *, significance: float = 0.05) -> bool:
        """TODO docs"""


class ActivityProfile(Profile):
    """"TODO docs"""

    activities: typing.Iterable[Activity]
    resources: typing.Iterable[Resource]
    # requirements
    resource_frequency: typing.MutableMapping[tuple[Activity, Resource], int]
    # demand
    demand: typing.MutableMapping[Activity, float]
    # behaviour
    arrival_distribution: typing.MutableMapping[Activity, Calendar]
    # complexity
    complexity_deviation: typing.MutableMapping[Activity, float]
    # interactions
    co_occurrence_index: typing.MutableMapping[tuple[Activity, Activity], int]

    def __init__(
            self: typing.Self,
            activities: typing.Iterable[Activity],
            resources: typing.Iterable[Resource],
            resource_frequency: typing.MutableMapping[tuple[Activity, Resource], int],
            demand: typing.MutableMapping[Activity, float],
            arrival_distribution: typing.MutableMapping[Activity, Calendar],
            complexity_deviation: typing.MutableMapping[Activity, float],
            co_occurrence_index: typing.MutableMapping[tuple[Activity, Activity], int],
    ) -> None:
        self.activities = activities
        self.resources = resources
        self.resource_frequency = resource_frequency
        self.demand = demand
        self.arrival_distribution = arrival_distribution
        self.complexity_deviation = complexity_deviation
        self.co_occurrence_index = co_occurrence_index

    @profile()
    def statistically_equals(self: typing.Self, other: ActivityProfile, *, significance: float = 0.05) -> bool:
        """TODO docs"""
        # compute the sets of all activities and resources present in any of the profiles
        all_activities = set(itertools.chain(self.activities, other.activities))
        all_resources = set(itertools.chain(self.resources, other.resources))

        # compute total counts for each activity for both self and other
        self_activity_totals = {
            activity: sum(value for ((act, _), value) in self.resource_frequency.items() if act == activity)
            for activity in self.activities
        }
        other_activity_totals = {
            activity: sum(value for ((act, _), value) in other.resource_frequency.items() if act == activity)
            for activity in other.activities
        }

        # check for drifts in the resource frequency using a poisson E-test
        resource_frequency_drift = {}
        for (activity, resource) in itertools.product(all_activities, all_resources):
            resource_frequency_drift[activity, resource] = scipy.stats.poisson_means_test(
                # the first param is the number of instances of activity executed by resource in self model
                self.resource_frequency[activity, resource],
                # the second param is the total number of instances of activity in self model
                self_activity_totals[activity],
                # the third param is the number of instances of activity executed by resource in other model
                other.resource_frequency[activity, resource],
                # the forth param is the total number of instances of activity in other model
                other_activity_totals[activity],
            )

        # TODO complete

        return all(result.pvalue >= significance for result in resource_frequency_drift.values())

    @staticmethod
    @profile()
    def discover(log: Log) -> ActivityProfile:
        """TODO docs"""
        activities = {event.activity for event in log}
        resources = {event.resource for event in log}
        activity_profile = ActivityProfile(
            activities=activities,
            resources=resources,
            resource_frequency=defaultdict(lambda: 0),
            demand=defaultdict(lambda: 0.0),
            arrival_distribution=defaultdict(Calendar),
            complexity_deviation=defaultdict(lambda: 0.0),
            co_occurrence_index=defaultdict(lambda: 0),
        )

        for activity in activities:
            activity_instances = [event for event in log if event.activity == activity]

            # compute resource frequency for each activity
            for resource in resources:
                activity_profile.resource_frequency[(activity, resource)] = len([event for event in activity_instances if event.resource == resource])

            # compute the workforce demand
            required_work = sum([event.processing_time.effective.duration for event in activity_instances], timedelta())
            total_work = sum([event.processing_time.effective.duration for event in log], timedelta())
            activity_profile.demand[activity] = required_work / total_work

            # compute the arrival distribution
            activity_profile.arrival_distribution[activity] = Calendar.discover(activity_instances)

            # compute the complexity deviation
            mean_self_time = sum([event.processing_time.effective.duration for event in activity_instances], timedelta()) / len(activity_instances)
            mean_time = sum([event.processing_time.effective.duration for event in log], timedelta()) / len(list(log))
            activity_profile.complexity_deviation[activity] = mean_self_time / mean_time

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

    activities: typing.Iterable[Activity]
    resources: typing.Iterable[Resource]
    # skills
    instance_count: typing.MutableMapping[tuple[Resource, Activity], int]
    # utilization
    utilization_index: typing.MutableMapping[Resource, float]
    # preferences
    effort_distribution: typing.MutableMapping[Resource, Calendar]
    # productivity
    performance_deviation: typing.MutableMapping[tuple[Resource, Activity], float]
    # collaboration
    collaboration_index: typing.MutableMapping[tuple[Resource, Resource], int]

    def __init__(
            self: typing.Self,
            activities: typing.Iterable[Activity],
            resources: typing.Iterable[Resource],
            instance_count: typing.MutableMapping[tuple[Resource, Activity], int],
            utilization_index: typing.MutableMapping[Resource, float],
            effort_distribution: typing.MutableMapping[Resource, Calendar],
            performance_deviation: typing.MutableMapping[tuple[Resource, Activity], float],
            collaboration_index: typing.MutableMapping[tuple[Resource, Resource], int],
    ) -> None:
        self.activities = activities
        self.resources = resources
        self.instance_count = instance_count
        self.utilization_index = utilization_index
        self.effort_distribution = effort_distribution
        self.performance_deviation = performance_deviation
        self.collaboration_index = collaboration_index

    @profile()
    def statistically_equals(self: typing.Self, other: ResourceProfile, *, significance: float = 0.05) -> bool:
        """TODO docs"""
        # compute the sets of all activities and resources present in any of the profiles
        all_activities = set(itertools.chain(self.activities, other.activities))
        all_resources = set(itertools.chain(self.resources, other.resources))

        # check instance count
        instance_count_drift = {}
        for (resource, activity) in itertools.product(all_resources, all_activities):
            instance_count_drift[(resource, activity)] = scipy.stats.poisson_means_test(
                # check the count of instances for an activity in self
                self.instance_count[(resource, activity)],
                # wrt. the total instances executed by the resource in self, vs
                sum(value for ((_resource, _), value) in self.instance_count.items() if _resource == resource),
                # the count of instances for an activity in other
                other.instance_count[(resource, activity)],
                # wrt. the total instances executed by the resource in other
                sum(value for ((_resource, _), value) in other.instance_count.items() if _resource == resource),
            )

        # check utilization index
        utilization_index_drift = {}
        for resource in all_resources:
            utilization_index_drift[resource] = scipy.stats.poisson_means_test(
                # check the percentage of time the resource is busy in self, vs
                int(self.utilization_index[resource] * 100),
                100,
                # the percentage of time the resource is busy in other
                int(other.utilization_index[resource] * 100),
                100,
            )

        # check effort distribution
        effort_distribution_drift = {}
        for resource in all_resources:
            effort_distribution_drift[resource] = self.effort_distribution[resource].statistically_equals(
                other.effort_distribution[resource], significance=significance,
            )

        # check performance deviation
        performance_deviation_drift = {}
        for (resource, activity) in itertools.product(all_resources, all_activities):
            performance_deviation_drift[(resource, activity)] = scipy.stats.poisson_means_test(
                # check the deviation of one activity (in percentage over the mean) in self, vs
                int(self.performance_deviation[(resource, activity)] * 100),
                100,
                # the deviation of one activity (in percentage over the mean) in other
                int(other.performance_deviation[(resource, activity)] * 100),
                100,
            )

        # check collaboration index
        collaboration_index_drift = {}
        for (resource1, resource2) in itertools.combinations(all_resources, 2):
            collaboration_index_drift[(resource1, resource2)] = scipy.stats.poisson_means_test(
                # check the count of collaborations in self
                self.collaboration_index[(resource1, resource2)],
                # wrt. the sum of cases executed by resource 1 or resource 2 in self, vs
                self.collaboration_index[(resource1, resource1)] + self.collaboration_index[(resource2, resource2)],
                # the count of instances for an activity in other
                other.collaboration_index[(resource1, resource2)],
                # wrt. the total instances executed by the resource in other
                other.collaboration_index[(resource1, resource1)] + other.collaboration_index[(resource2, resource2)],
            )

        return (
                all(value.pvalue >= significance for value in instance_count_drift.values()) and
                all(value.pvalue >= significance for value in utilization_index_drift.values()) and
                all(effort_distribution_drift.values()) and
                all(value.pvalue >= significance for value in performance_deviation_drift.values()) and
                all(value.pvalue >= significance for value in collaboration_index_drift.values())
        )

    @staticmethod
    @profile()
    def discover(log: Log) -> ResourceProfile:
        """TODO docs"""
        activities = {event.activity for event in log}
        resources = {event.resource for event in log}
        resource_profile = ResourceProfile(
            activities=activities,
            resources=resources,
            instance_count=defaultdict(lambda: 0),
            utilization_index=defaultdict(lambda: 0.0),
            effort_distribution=defaultdict(Calendar),
            performance_deviation=defaultdict(lambda: 0.0),
            collaboration_index=defaultdict(lambda: 0),
        )

        for resource in resources:
            events_by_resource = [event for event in log if event.resource == resource]

            # compute activity instance count for each activity executed by the resource
            for activity in activities:
                resource_profile.instance_count[(resource, activity)] = len([event for event in events_by_resource if event.activity == activity])

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
            for activity in activities:
                all_events = [event for event in log if event.activity == activity]
                self_events = [event for event in events_by_resource if event.activity == activity]
                # compute mean execution time for all events and self events
                mean_self_time = sum([event.processing_time.effective.duration for event in self_events], timedelta()) / len(self_events)
                mean_execution_time = sum([event.processing_time.effective.duration for event in all_events], timedelta()) / len(all_events)
                # the deviation is the factor of self time vs total time (<1.0 means over performance, >1.0 means under performance)
                resource_profile.performance_deviation[(resource, activity)] = mean_self_time / mean_execution_time

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
