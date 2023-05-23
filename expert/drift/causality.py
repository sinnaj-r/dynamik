from __future__ import annotations

import enum
import typing
from datetime import timedelta
from statistics import mean

import scipy
from anytree import Node

from expert.drift.features import DriftFeatures, Pair
from expert.model import Test
from expert.utils.activities import compute_activity_batch_sizing, compute_prioritized_activities


class CAUSE_DETAILS_TYPE(enum.Enum):
    """The type of the details object for the tree nodes"""

    SIZE_SUMMARY_PAIR_PER_ACTIVITY = 0
    DURATION_SUMMARY_PAIR = 1
    DURATION_SUMMARY_PAIR_PER_ACTIVITY = 2
    DURATION_SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE = 3
    FREQUENCY_SUMMARY_PAIR_PER_ACTIVITY = 4
    DIFFERENCE_PER_ACTIVITY = 5
    CALENDAR_PER_RESOURCE = 6


def default_drift_causality_test_factory(
        alpha: float = 0.05,
        min_diff: timedelta = timedelta(minutes=0),
) -> Test:
    """Default statistical test factory used for comparing the reference and running distributions for drift causes"""
    def __test_causes(
            reference: typing.Iterable[typing.Any],
            running: typing.Iterable[typing.Any],
            *,
            alternative: str = "less",
    ) -> bool:
        # transform timedeltas to floats for applying the test
        if isinstance(list(reference)[0], timedelta):
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


def __check_contention_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: Test,
) -> None:
    # check contention times for common activities
    activities = set(drift_features.contention_time.reference.keys())\
        .intersection(set(drift_features.contention_time.running.keys()))
    # compute batch sizes for running and reference data
    reference_times = drift_features.contention_time.reference
    running_times = drift_features.contention_time.running
    # check if any batch size changed between reference and running
    drifting_activities = {
        activity: Pair(
            reference=scipy.stats.describe([
                time.total_seconds() for time in reference_times[activity]
            ]),
            running=scipy.stats.describe([
                time.total_seconds() for time in running_times[activity]
            ]),
            unit="duration",
        )
        for activity in activities
        if test(reference_times[activity], running_times[activity])
    }
    # if there are significant changes, add a new node to the cause tree
    if len(drifting_activities) > 0:
        node = Node(
            "contention time increased",
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
            parent=parent,
        )
        __check_arrival_rate(drift_features, node, test)


def __check_batch_sizes(
        drift_features: DriftFeatures,
        activities: typing.Iterable[str],
        parent: Node,
        test: Test,
) -> None:
    # compute batch sizes for running and reference data
    reference_sizes = compute_activity_batch_sizing(drift_features.model.reference_model)
    running_sizes = compute_activity_batch_sizing(drift_features.model.running_model)
    # check if any batch size changed between reference and running
    drifting_sizes = {
        activity: Pair(
            reference=scipy.stats.describe(reference_sizes[activity]),
            running=scipy.stats.describe(running_sizes[activity]),
            unit = "size",
        )
        for activity in activities
        if test(reference_sizes[activity], running_sizes[activity])
    }
    # if there are significant changes, add a new node to the cause tree
    if len(drifting_sizes) > 0:
        Node(
            "batch size increased",
            details = drifting_sizes,
            type=CAUSE_DETAILS_TYPE.SIZE_SUMMARY_PAIR_PER_ACTIVITY,
            parent=parent,
        )


def __check_batching_policy(
        drift_features: DriftFeatures,
        activities: typing.Iterable[str],
        parent: Node,
        test: Test,
) -> None:
    __check_batch_sizes(drift_features, activities, parent, test)


def __check_arrival_rate_decrease_for_activities(
        drift_features: DriftFeatures,
        activities: typing.Iterable[str],
        parent: Node,
        test: Test,
) -> None:
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
        if test(drift_features.arrival_rate.running[activity], drift_features.arrival_rate.reference[activity])
    }
    # if there are increased arrival rates for any activity, add a node to the causes tree
    if len(drifting_activities) > 0:
        Node(
            "arrival rate decreased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
        )


def __check_batching_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: Test,
) -> None:
    activities = set(
        list(drift_features.batching_time.reference.keys()) + list(drift_features.batching_time.running.keys()),
        )
    # check which activities present a drift in the batching time
    drifting_activities = {
        # save the summary for the drifting activities
        activity: Pair(
            reference=scipy.stats.describe([
                time.total_seconds() for time in drift_features.batching_time.reference[activity]
            ]),
            running=scipy.stats.describe([
                time.total_seconds() for time in drift_features.batching_time.running[activity]
            ]),
            unit=drift_features.batching_time.unit,
        )
        for activity in activities
        if test(drift_features.batching_time.reference[activity], drift_features.batching_time.running[activity])
    }

    # check batching times
    if len(drifting_activities) > 0:
        node = Node(
            "batching time increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
        )
        # if there are changes, check for drifts in the batching policy
        __check_batching_policy(
            drift_features,
            drifting_activities.keys(),
            node,
            test,
        )
        # if changed, check arrival rates (lower arrival rates can lead to higher batching times)
        __check_arrival_rate_decrease_for_activities(
            drift_features,
            drifting_activities.keys(),
            node,
            test,
        )


def __check_priorities(
        drift_features: DriftFeatures,
        activities: typing.Iterable[str],
        parent: Node,
) -> None:
    # compute the priorities in reference
    reference_priorities = compute_prioritized_activities(drift_features.model.reference_model)
    # compute the priorities in running
    running_priorities = compute_prioritized_activities(drift_features.model.running_model)
    # check new priorities in running
    priorities_added = {
        activity: running_priorities[activity] - reference_priorities[activity]
        for activity in activities
        if len(running_priorities[activity] - reference_priorities[activity]) > 0
    }
    # if there are new priorities, add a node with the details
    if len(priorities_added) > 0:
        Node(
            "new priorities causing longer waiting times",
            parent=parent,
            details=priorities_added,
            type=CAUSE_DETAILS_TYPE.DIFFERENCE_PER_ACTIVITY,
        )

    # check removed priorities in running
    priorities_removed = {
        activity: reference_priorities[activity] - running_priorities[activity]
        for activity in activities
        if len(reference_priorities[activity] - running_priorities[activity]) > 0
    }
    # if there are removed priorities, add a node with the details
    if len(priorities_removed) > 0:
        Node(
            "removed priorities causing longer waiting times",
            parent=parent,
            details=priorities_removed,
            type=CAUSE_DETAILS_TYPE.DIFFERENCE_PER_ACTIVITY,
        )


def __check_prioritization_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: Test,
) -> None:
    # get prioritization times
    reference_prioritization_times = drift_features.prioritization_time.reference
    running_prioritization_times = drift_features.prioritization_time.running
    # get the set of activities
    activities = set(list(reference_prioritization_times.keys()) + list(running_prioritization_times.keys()))
    # check which activities present a drift in the prioritization times
    drifting_activities = {
        # save the summary for the drifting activities
        activity: Pair(
            reference=scipy.stats.describe([
                time.total_seconds() for time in reference_prioritization_times[activity]
            ]),
            running=scipy.stats.describe([
                time.total_seconds() for time in running_prioritization_times[activity]
            ]),
            unit=drift_features.prioritization_time.unit,
        )
        for activity in activities
        if test(reference_prioritization_times[activity], running_prioritization_times[activity])
    }
    # if there are any drift, add a node to the causes tree with the causes
    if len(drifting_activities) > 0:
        node = Node(
            "prioritization time increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
        )
        # check how priorities changed
        __check_priorities(drift_features, drifting_activities.keys(), node)


def __check_extraneous_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: Test,
) -> None:
    # get extraneous times
    reference_extraneous_times = drift_features.extraneous_time.reference
    running_extraneous_times = drift_features.extraneous_time.running
    # get the set of activities
    activities = set(list(reference_extraneous_times.keys()) + list(running_extraneous_times.keys()))
    # check which activities present a drift in the extraneous times
    drifting_activities = {
        # save the summary for the drifting activities
        activity: Pair(
            reference=scipy.stats.describe([
                time.total_seconds() for time in reference_extraneous_times[activity]
            ]),
            running=scipy.stats.describe([
                time.total_seconds() for time in running_extraneous_times[activity]
            ]),
            unit=drift_features.extraneous_time.unit,
        )
        for activity in activities
        if test(reference_extraneous_times[activity], running_extraneous_times[activity])
    }
    # if there are any drift, add a node to the causes tree with the causes
    if len(drifting_activities) > 0:
        Node(
            "extraneous time increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
        )

def __check_arrival_rate(
        drift_features: DriftFeatures,
        parent: Node,
        test: Test,
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
    # if there are increased arrival rates for any activity, add a node to the causes tree
    if len(drifting_activities) > 0:
        Node(
            "arrival rate increased",
            parent=parent,
            details=drifting_activities,
            type=CAUSE_DETAILS_TYPE.FREQUENCY_SUMMARY_PAIR_PER_ACTIVITY,
        )


def __check_resources_underperforming_in_running(
        drift_features: DriftFeatures,
        resources_allocation_per_activity: typing.Mapping[str, set[str]],
        parent: Node,
        test: Test,
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
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE,
        )


def __check_resources_overperforming_in_reference(
        drift_features: DriftFeatures,
        resources_allocation_per_activity: typing.Mapping[str, set[str]],
        parent: Node,
        test: Test,
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
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY_AND_RESOURCE,
        )


def __check_resources_allocation(
        drift_features: DriftFeatures,
        activities: set[str],
        parent: Node,
        test: Test,
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
        test: Test,
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
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
        )

        # check for changes in the batching times
        __check_batching_times(drift_features, node, test)
        # check for changes in the contention times
        __check_contention_times(drift_features, node, test)
        # check for changes in the prioritization times
        __check_prioritization_times(drift_features, node, test)
        # check for changes in extraneous times
        # __check_extraneous_times(drift_features, node, test)

def __check_execution_times(
        drift_features: DriftFeatures,
        parent: Node,
        test: Test,
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
            type=CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR_PER_ACTIVITY,
        )

        __check_resources_allocation(drift_features, set(drifting_activities.keys()), node, test)


def explain_drift(
        drift_features: DriftFeatures,
        test: Test = default_drift_causality_test_factory(),
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
            type = CAUSE_DETAILS_TYPE.DURATION_SUMMARY_PAIR,
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
