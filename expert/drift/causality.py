from __future__ import annotations

import enum
import typing
from datetime import timedelta
from statistics import mean

import scipy
from anytree import Node

from expert.drift.features import DriftFeatures, Pair


class CAUSE_DETAILS_TYPE(enum.Enum):
    """The type of the details object for the tree nodes"""

    SUMMARY_PAIR = 1
    SUMMARY_PAIR_PER_ACTIVITY = 2
    DIFFERENCE_PER_ACTIVITY = 3
    SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE = 4
    CALENDAR_PER_RESOURCE = 5


def default_drift_causality_test_factory(alpha: float = 0.05, min_diff: timedelta = timedelta(minutes=0)) -> \
        typing.Callable[[typing.Sequence[typing.Any], typing.Sequence[typing.Any]], bool]:
    """Default statistical test factory used for comparing the reference and running distributions for drift causes"""
    def __test_causes(
            reference: typing.Sequence[typing.Any],
            running: typing.Sequence[typing.Any],
            alternative: str = "less",
    ) -> bool:
        # transform timedeltas to floats for applying the test
        if isinstance(reference[0], timedelta):
            reference = [value.total_seconds() for value in reference]
            running = [value.total_seconds() for value in running]

            # compute the mean times
            reference_mean = timedelta(seconds=mean(reference))
            running_mean = timedelta(seconds=mean(running))

            # if the difference is lower than the threshold, return false
            if (running_mean - reference_mean) < min_diff:
                return False

        # save test result
        test = scipy.stats.mannwhitneyu(reference, running, alternative=alternative)

        # check the test
        return test.pvalue < alpha

    return __test_causes


def __check_batching_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    if test(drift_features.batching_time.reference, drift_features.batching_time.running):
        Node(
            "Batching time",
            parent=parent,
        )


def __check_contention_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    if test(drift_features.contention_time.reference, drift_features.contention_time.running):
        Node(
            "Contention time",
            parent=parent,
        )


def __check_prioritization_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    if test(drift_features.prioritization_time.reference, drift_features.prioritization_time.running):
        Node(
            "Prioritization time",
            parent=parent,
        )


def __check_arrival_rate(
        drift_features: DriftFeatures,
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    # get the list of activities
    activities = set(
        list(drift_features.waiting_time.reference.keys()) + list(drift_features.waiting_time.running.keys()),
        )

    # build a map of drifting activities and their pre- and post-drift waiting time distributions to describe the drift
    drifting_activities = {
        # the pair activity: description
        activity: Pair(
            # the description of the reference distribution
            reference=scipy.stats.describe(drift_features.arrival_rate.reference[activity]),
            # the description of the running distribution
            running=scipy.stats.describe(drift_features.arrival_rate.running[activity]),
            unit=drift_features.arrival_rate.unit,
        )
        # for each activity in the set of activities
        for activity in activities
        # if there is a drift between reference and running data
        if test(drift_features.arrival_rate.reference[activity], drift_features.arrival_rate.running[activity])
    }

    if len(drifting_activities) > 0:
        Node(
            "arrival rate increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY,
        )


def __check_resources_underperforming_in_running(
        drift_features: DriftFeatures,
        resources_allocation_per_activity: typing.Mapping[str, set[str]],
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    # get the reference execution times (already computed in drift features container)
    execution_times_reference: typing.Mapping[str, typing.Iterable[timedelta]] = drift_features.execution_time.reference
    # compute the execution times per activity and resource for the new resources
    execution_times_running_per_resource: typing.Mapping[str, typing.Mapping[str, typing.Iterable[timedelta]]] = {
        activity: {
            resource: [
                event.execution_time for event in drift_features.model.running_model if event.resource == resource
            ]  for resource in resources_allocation_per_activity[activity]
        } for activity in resources_allocation_per_activity
    }
    # check which resource allocations are under-performing for each activity
    underperforming_resources_per_activity = {
        activity: {
            resource: Pair(
                reference=scipy.stats.describe([
                    time.total_seconds() for time in execution_times_reference[activity]
                ]),
                running=scipy.stats.describe([
                    time.total_seconds() for time in execution_times_running_per_resource[activity][resource]
                ]),
                unit="duration",
            )
            for resource in resources_allocation_per_activity[activity]
            if test(execution_times_reference[activity], execution_times_running_per_resource[activity][resource])
        } for activity in resources_allocation_per_activity
    }

    # clean empty entries
    underperforming_resources_per_activity = {
        activity: resources
        for (activity, resources) in underperforming_resources_per_activity.items()
        if len(resources) > 0
    }

    # create the node describing this drift and add it to the causes tree
    if len(underperforming_resources_per_activity) > 0:
        Node(
            "resources under-performing",
            parent=parent,
            details=underperforming_resources_per_activity,
            type=CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE,
        )


def __check_resources_overperforming_in_reference(
        drift_features: DriftFeatures,
        resources_allocation_per_activity: typing.Mapping[str, set[str]],
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    # get the running execution times (already computed in drift features container)
    execution_times_running: typing.Mapping[str, typing.Iterable[timedelta]] = drift_features.execution_time.running
    # compute the execution times per activity and resource for the reference model
    execution_times_reference: typing.Mapping[str, typing.Mapping[str, typing.Iterable[timedelta]]] = {
        activity: {
            resource: [
                event.execution_time for event in drift_features.model.reference_model if event.resource == resource
            ]  for resource in resources_allocation_per_activity[activity]
        } for activity in resources_allocation_per_activity
    }
    # check which resource allocations were over-performing for each activity
    overperforming_resources_per_activity = {
        activity: {
            resource: Pair(
                reference=scipy.stats.describe([
                    time.total_seconds() for time in execution_times_reference[activity][resource]
                ]),
                running=scipy.stats.describe([time.total_seconds() for time in execution_times_running[activity]]),
                unit="duration",
            )
            for resource in resources_allocation_per_activity[activity]
            if test(execution_times_reference[activity][resource], execution_times_running[activity])
        } for activity in resources_allocation_per_activity
    }
    # clean empty entries
    overperforming_resources_per_activity = {
        activity: resources
        for (activity, resources) in overperforming_resources_per_activity.items()
        if len(resources) > 0
    }

    # create the node describing this drift and add it to the causes tree
    if len(overperforming_resources_per_activity) > 0:
        Node(
            "resources over-performing",
            parent=parent,
            details=overperforming_resources_per_activity,
            type=CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE,
        )


def __check_resources_allocation(
        drift_features: DriftFeatures,
        activities: set[str],
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    # compute resources allocations for each activity
    reference_allocations = drift_features.resources_allocation.reference
    running_allocations = drift_features.resources_allocation.running

    # compute the added, removed and common resources for the activities
    new_allocations = {}
    removed_allocations = {}
    common_allocations = {}

    for activity in activities:
        # compute the new allocations (that is, the difference between the running allocations and the reference ones)
        new_allocations[activity] = running_allocations[activity] - reference_allocations[activity]
        # compute the removed allocations (the difference between the reference and the running allocations)
        removed_allocations[activity] = reference_allocations[activity] - running_allocations[activity]
        # compute the common allocations (the intersection between the reference and the running allocations)
        common_allocations[activity] = reference_allocations[activity].intersection(running_allocations[activity])

    # if there are common allocations, compute their performance
    if any(len(allocation) > 0 for allocation in common_allocations.values()):
        allocations = {act: res for (act, res) in common_allocations.items() if len(res) > 0}
        # check for changes in common resources availability
        __check_resources_availability(drift_features, allocations, parent)
        # check for changes in common resources performance
        __check_resources_underperforming_in_running(drift_features, allocations, parent, test)

    # if there are new allocations, add a new node to the causes tree
    if any(len(allocation) > 0 for allocation in new_allocations.values()):
        allocations = { act: res for (act, res) in new_allocations.items() if len(res) > 0 }
        node = Node(
            "new resource allocations",
            details=allocations,
            type=CAUSE_DETAILS_TYPE.DIFFERENCE_PER_ACTIVITY,
        )
        __check_resources_underperforming_in_running(drift_features, allocations, node, test)

        if len(node.children) > 0:
            node.parent = parent

    # if there are removed allocations, add a new node to the causes tree
    if any(len(allocation) > 0 for allocation in removed_allocations.values()):
        allocations = {act: res for (act, res) in removed_allocations.items() if len(res) > 0}
        node = Node(
            "removed resource allocations",
            details={ act: res for (act, res) in removed_allocations.items() if len(res) > 0 },
            type=CAUSE_DETAILS_TYPE.DIFFERENCE_PER_ACTIVITY,
        )
        __check_resources_overperforming_in_reference(drift_features, allocations, node, test)

        if len(node.children) > 0:
            node.parent = parent


def __check_resources_availability(
        drift_features: DriftFeatures,
        allocations: typing.Mapping[str, typing.Iterable[str]],
        parent: Node,
) -> None:
    resources = { resource for resources in allocations.values() for resource in resources }

    # keep the availability intervals present in reference and missing in running
    removed = {
        resource: set(drift_features.resources_availability.reference[resource]) - set(drift_features.resources_availability.running[resource])
        for resource in resources
        if len(set(drift_features.resources_availability.reference[resource]) - set(drift_features.resources_availability.running[resource])) > 0
    }

    if len(removed) > 0:
        Node(
            "removed resources availability slots",
            details=removed,
            type=CAUSE_DETAILS_TYPE.CALENDAR_PER_RESOURCE,
            parent=parent,
        )


def __check_waiting_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    # get the list of activities
    activities = set(
        list(drift_features.waiting_time.reference.keys()) + list(drift_features.waiting_time.running.keys()),
        )

    # build a map of drifting activities and their pre- and post-drift waiting time distributions to describe the drift
    drifting_activities = {
        # the pair activity: description
        activity: Pair(
            # the description of the reference distribution
            reference=scipy.stats.describe([
                value.total_seconds() if isinstance(value, timedelta) else value
                for value in drift_features.waiting_time.reference[activity]
            ]),
            # the description of the running distribution
            running=scipy.stats.describe([
                value.total_seconds() if isinstance(value, timedelta) else value
                for value in drift_features.waiting_time.running[activity]
            ]),
            unit=drift_features.waiting_time.unit,
        )
        # for each activity in the set of activities
        for activity in activities
        # if there is a drift between reference and running data
        if test(drift_features.waiting_time.reference[activity], drift_features.waiting_time.running[activity])
    }

    if len(drifting_activities) > 0:
        node = Node(
            "waiting time increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY,
        )

        # check for changes in the batching times
        # __check_batching_times(drift_features, node, test)
        # check for changes in the contention times
        # __check_contention_times(drift_features, node, test)
        # check for changes in the prioritization times
        # __check_prioritization_times(drift_features, node, test)
        # check for changes in the arrival rates
        __check_arrival_rate(drift_features, node, test)


def __check_execution_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool],
) -> None:
    # Get the list of activities
    activities=set(
        list(drift_features.execution_time.reference.keys()) + list(drift_features.execution_time.running.keys()),
        )

    # Build a map of drifting activities and their pre- and post-drift execution duration distributions to describe the
    # change
    drifting_activities = {
        # the pair activity: description
        activity: Pair(
            # the description of the reference distribution
            reference=scipy.stats.describe([
                value.total_seconds() if isinstance(value, timedelta) else value
                for value in drift_features.execution_time.reference[activity]
            ]),
            # the description of the running distribution
            running=scipy.stats.describe([
                value.total_seconds() if isinstance(value, timedelta) else value
                for value in drift_features.execution_time.running[activity]
            ]),
            unit=drift_features.execution_time.unit,
        )
        # for each activity in the set of activities
        for activity in activities
        # if there is a drift between reference and running data
        if test(drift_features.execution_time.reference[activity], drift_features.execution_time.running[activity])
    }

    # if at least one activity presents a drift, add the node to the causes tree
    if len(drifting_activities) > 0:
        node = Node(
            "execution time increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.SUMMARY_PAIR_PER_ACTIVITY,
        )

        __check_resources_allocation(drift_features, set(drifting_activities.keys()), node, test)


def explain_drift(
        drift_features: DriftFeatures,
        test: typing.Callable[[typing.Iterable[typing.Any], typing.Iterable[typing.Any]], bool] = default_drift_causality_test_factory(),
) -> Node:
    """Build a tree with the causes that explain the drift characterized by the given drift features"""
    # if there are a drift in the cycle time distribution, build a node
    if test(drift_features.case_duration.reference, drift_features.case_duration.running):
        root = Node(
            "cycle time increased",
            details=Pair(
                reference=scipy.stats.describe(drift_features.case_duration.reference),
                running=scipy.stats.describe(drift_features.case_duration.running),
                unit=drift_features.case_duration.unit,
            ),
            type = CAUSE_DETAILS_TYPE.SUMMARY_PAIR,
        )

        # if there is a drift, check the waiting times
        __check_waiting_times(drift_features, root, test)
        # if there is a drift, check the execution times
        __check_execution_times(drift_features, root, test)
    # if there are no drift in the cycle times, return a new node with an unknown cause
    else:
        return Node("Unknown")

    # return the root node for the causes tree
    return root
