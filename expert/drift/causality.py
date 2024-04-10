from __future__ import annotations

import itertools
import typing
from datetime import timedelta

import pandas as pd
import scipy
from anytree import AnyNode
from intervaltree import IntervalTree
from pandas import CategoricalDtype

from expert.drift.model import Drift, Pair
from expert.model import Event
from expert.timer import profile
from expert.utils.batching import (
    build_batch_creation_features,
    build_batch_firing_features,
    discover_batch_creation_policies,
    discover_batch_firing_policies,
)
from expert.utils.calendars import compute_weekly_available_time_per_resource, discover_calendars
from expert.utils.cases import compute_cases_length, compute_inter_arrival_times
from expert.utils.feature_selection import chained_selectors, from_model, select_relevant_features, univariate
from expert.utils.prioritization import build_prioritization_features, discover_prioritization_policies
from expert.utils.rules import Rule, compute_rule_score, filter_log
from expert.utils.statistical_tests import categorical_test, continuous_test


def __check_policy(
        policy: Rule,
        reference_features: pd.DataFrame,
        running_features: pd.DataFrame,
        *,
        samples: int = 12,
        sample_size: float = 0.8,
        policy_type: str = "",
) -> AnyNode | None:
    # compute the rule score for the samples, both for reference and running data
    reference_score = compute_rule_score(
        policy,
        data=reference_features,
        class_attr=policy.training_class,
        n_samples=samples,
        sample_size=sample_size,
    )
    running_score = compute_rule_score(
        policy,
        data=running_features,
        class_attr=policy.training_class,
        n_samples=samples,
        sample_size=sample_size,
    )

    # check if scores are equal
    if continuous_test([score.f1_score for score in reference_score], [score.f1_score for score in running_score]):
        return AnyNode(
            # what changed? the score for the prioritization policy
            what=f"score for {policy_type} policy '{policy.__repr__()}' changed!",
            # how did it change? include the scores for before and after the change
            how=Pair(
                reference=scipy.stats.describe([score.f1_score for score in reference_score]),
                running=scipy.stats.describe([score.f1_score for score in running_score]),
                unit="score",
            ),
            # data contains the raw scores for the CV evaluation of the policy
            data=Pair(
                reference=reference_score,
                running=running_score,
                unit="score",
            ),
        )

    # if no change is found in the policy score, return None
    return None


def __check_attributes(drift: Drift, *, class_extractor: typing.Callable[[Event], float]) -> list[AnyNode]:
    # cases complexity could be assessed by the attributes in the events
    # build a list that will contain the causes of the drift
    causes = []
    # first, find which attributes have an impact on the effective processing time
    selected_reference_features, reference_features = select_relevant_features(
        drift.reference_model,
        class_extractor=class_extractor,
        predictors_extractor=lambda event: event.attributes,
        feature_selector=chained_selectors([univariate(), from_model()]),
    )
    selected_running_features, running_features = select_relevant_features(
        drift.running_model,
        class_extractor=class_extractor,
        predictors_extractor=lambda event: event.attributes,
        feature_selector=chained_selectors([univariate(), from_model()]),
    )
    # then, for each feature that has an impact in the effective processing time, check if there are significant
    # differences in the values distribution for both pre- and post-change
    for feature in {*selected_reference_features, *selected_running_features}:
        # if the feature is present in both pre- and post- change data, compare their distributions
        if feature in reference_features.columns and feature in running_features.columns:
            # if the feature is categorical and there are differences in the histograms, add a subtree to the causes of the drift
            if isinstance(reference_features[feature].dtype, CategoricalDtype) and categorical_test(
                    reference_features[feature].tolist(), running_features[feature].tolist()):
                causes.append(
                    AnyNode(
                        # what changed? the distribution of values of a given feature
                        what=f"significant differences on the distribution of values for attribute '{feature}'",
                        # how did it change? include the histograms for before and after the change
                        how=Pair(
                            reference=reference_features[feature].value_counts(normalize=True, sort=False).to_dict(),
                            running=running_features[feature].value_counts(normalize=True, sort=False).to_dict(),
                            unit="histogram",
                        ),
                        # data contains the raw data used for building the histograms
                        data=Pair(
                            reference=drift.reference_model,
                            running=drift.running_model,
                            unit="events",
                        ),
                    ),
                )
            # if the feature is numerical, compare the distribution of values and add a subtree with the differences
            elif continuous_test(reference_features[feature].tolist(), running_features[feature].tolist()):
                causes.append(
                    AnyNode(
                        # what changed? the distribution of values of a given feature
                        what=f"significant differences on the distribution of values for attribute '{feature}'",
                        # how did it change? include the histograms for before and after the change
                        how=Pair(
                            reference=scipy.stats.describe(reference_features[feature]),
                            running=scipy.stats.describe(running_features[feature]),
                        ),
                        # data contains the raw data used for building the histograms
                        data=Pair(
                            reference=drift.reference_model,
                            running=drift.running_model,
                            unit="events",
                        ),
                    ),
                )
        # if the feature is only present before the drift, add a new node to the tree
        elif feature in reference_features.columns and feature not in running_features.columns:
            causes.append(
                AnyNode(
                    # what changed? the feature disappeared from the log
                    what=f"feature '{feature}' disappeared from the log",
                    # how did it change? include the histograms for before and after the change
                    how=Pair(
                        reference=reference_features[feature].value_counts(normalize=True, sort=False).to_dict(),
                        running=None,
                        unit="histogram",
                    ),
                    # data contains the raw data used for building the histograms
                    data=Pair(
                        reference=drift.reference_model,
                        running=drift.running_model,
                        unit="events",
                    ),
                ),
            )
        # if the feature is only visible after the drift, add a new node to the tree
        elif feature in running_features.columns and feature not in reference_features.columns:
            causes.append(
                AnyNode(
                    # what changed? a new feature appeared in the log
                    what=f"feature '{feature}' appeared in the log",
                    # how did it change? include the histograms for before and after the change
                    how=Pair(
                        reference=None,
                        running=running_features[feature].value_counts(normalize=True, sort=False).to_dict(),
                        unit="histogram",
                    ),
                    # data contains the raw data used for building the histograms
                    data=Pair(
                        reference=drift.reference_model,
                        running=drift.running_model,
                        unit="events",
                    ),
                ),
            )

    return causes


def __check_batching_policies(drift: Drift) -> list[AnyNode | None]:
    # build the batch creation features for both the reference and running models
    reference_creation_features = build_batch_creation_features(drift.reference_model)
    running_creation_features = build_batch_creation_features(drift.running_model)

    # build the batch firing features for both the reference and running models
    reference_firing_features = build_batch_firing_features(drift.reference_model)
    running_firing_features = build_batch_firing_features(drift.running_model)

    # extract the batch creation rules for both the reference and the running models
    reference_creation_policies = discover_batch_creation_policies(drift.reference_model)
    running_creation_policies = discover_batch_creation_policies(drift.running_model)

    # build a list to store the drift causes
    causes = []

    # check reference creation policies against reference and running models
    for creation_policy in reference_creation_policies:
        # check the batching creation policy
        causes.append(__check_policy(creation_policy, reference_creation_features, running_creation_features, policy_type="batch creation"))

        # after evaluating the creation policy, evaluate its firing policies
        # filter events in the current batch
        events_in_batch = filter_log(creation_policy, drift.reference_model)
        # discover the batch firing rules for the given creation policy
        firing_policies = discover_batch_firing_policies(events_in_batch)

        # check every reference firing rule score with running and reference models
        for firing_policy in firing_policies:
            causes.append(__check_policy(firing_policy, reference_firing_features, running_firing_features, policy_type=f"batch '{creation_policy.__repr__()}' firing"))

    # check running creation policies against reference and running
    for creation_policy in running_creation_policies:
        # check the batching creation policy
        causes.append(__check_policy(creation_policy, reference_creation_features, running_creation_features, policy_type="batch creation"))

        # after evaluating the creation policy, evaluate its firing policies
        # filter events in the current batch
        events_in_batch = filter_log(creation_policy, drift.running_model)
        # discover the batch firing rules for the given creation policy
        firing_policies = discover_batch_firing_policies(events_in_batch)

        # check every firing rule score with running and reference models
        for firing_policy in firing_policies:
            causes.append(__check_policy(firing_policy, reference_firing_features, running_firing_features, policy_type=f"batch '{creation_policy.__repr__()}' firing"))
    # return the list of drift causes
    return causes


def __check_prioritization_policies(drift: Drift) -> list[AnyNode | None]:
    # build the features for prioritization for evaluating the rules later
    reference_features = build_prioritization_features(drift.reference_model)
    running_features = build_prioritization_features(drift.running_model)

    # discover the rules for the reference and the running models
    reference_prioritization_policies = discover_prioritization_policies(drift.reference_model)
    running_prioritization_policies = discover_prioritization_policies(drift.running_model)

    # build a list of policies evaluations, that will contain the differences
    return [
        __check_policy(policy, reference_features, running_features, policy_type="prioritization") for policy in
        itertools.chain(reference_prioritization_policies, running_prioritization_policies)
    ]


def __check_weekly_available_hours(drift: Drift, *, granularity: timedelta = timedelta(hours=1)) -> AnyNode | None:
    # compute the total hours of availability per week both before and after the change
    reference_availability = sum(
        compute_weekly_available_time_per_resource(drift.reference_model, granularity).values(),
        start=timedelta(),
    )
    running_availability = sum(
        compute_weekly_available_time_per_resource(drift.running_model, granularity).values(),
        start=timedelta(),
    )

    # if they are different, report the change
    if reference_availability != running_availability:
        return AnyNode(
            # what changed? the availability working hours for the week (the weekly "capacity")
            what="weekly available working hours changed!",
            # how did it change? include the total number of hours of weekly availability for both before and after the change
            how=Pair(
                reference=reference_availability,
                running=running_availability,
                unit="hours",
            ),
            # data contains the raw data used to compute the availability hours per week
            data=Pair(
                reference=drift.reference_model,
                running=drift.running_model,
                unit="events",
            ),
        )
    # if no change is found, return None
    return None


def __check_case_length(drift: Drift) -> AnyNode | None:
    # compute the reference and running lengths for the cases
    reference_cases_length = compute_cases_length(drift.reference_model)
    running_cases_length = compute_cases_length(drift.running_model)

    # if the distribution of sizes is different, report the change
    if continuous_test(reference_cases_length, running_cases_length):
        return AnyNode(
            # what changed? the distribution of case lengths
            what="case length changed!",
            # how did it change? include the distribution of sizes for both pre- and post- change
            how=Pair(
                reference=scipy.stats.describe(reference_cases_length),
                running=scipy.stats.describe(running_cases_length),
                unit="size",
            ),
            # data contains the data use to run the test
            data=Pair(
                reference=reference_cases_length,
                running=running_cases_length,
                unit="size",
            ),
        )
    return None


def __check_inter_case_time(drift: Drift) -> AnyNode | None:
    # instead of computing the arrival rate, which implies defining a window for computing the frequency,
    # we compute the inter-case time, so we have a distribution of times between cases
    reference_inter_case_times = [round(time.total_seconds()) for time in
                                  compute_inter_arrival_times(drift.reference_model)]
    running_inter_case_times = [round(time.total_seconds()) for time in
                                compute_inter_arrival_times(drift.running_model)]

    # if the distributions are different, report the change
    if continuous_test(reference_inter_case_times, running_inter_case_times):
        return AnyNode(
            # what changed? the inter arrival time (the time between new cases arrive to the system)
            what="inter case arrival time changed",
            # how did it change? include the distributions for both before and after the change
            how=Pair(
                reference=scipy.stats.describe(reference_inter_case_times),
                running=scipy.stats.describe(running_inter_case_times),
                unit="seconds",
            ),
            # data contains the raw data used in the test
            data=Pair(
                reference=reference_inter_case_times,
                running=running_inter_case_times,
                unit="seconds",
            ),
        )
    # if no change detected, return None
    return None


# TODO CLEAN THIS METHOD
def __check_resources_availability(drift: Drift, *, granularity: timedelta = timedelta(hours=1)) -> list[AnyNode]:
    # compute resource calendars for both reference and running periods
    reference_calendars = discover_calendars(drift.reference_model, granularity)
    running_calendars = discover_calendars(drift.running_model, granularity)

    # create a list that will contain the changes in the resources availability
    causes = []

    # check every resource present in the model
    for resource in drift.resources:
        # resources can be added, removed or common but with different availability, so we treat each case separately
        # if a resource disappeared from the log
        if resource in reference_calendars and resource not in running_calendars:
            # resource is no longer available
            causes.append(
                AnyNode(
                    # what changed? the resource "resource" is not available after the drift
                    what=f"resource '{resource}' is no longer available",
                    # how changed? include the reference availability calendar for the resource
                    how=Pair(
                        reference=reference_calendars[resource],
                        running=None,
                        unit="calendar",
                    ),
                    # data contains the data used to discover the calendars
                    data=Pair(
                        reference=drift.reference_model,
                        running=drift.running_model,
                        unit="events",
                    ),
                    # include the granularity used for computing the calendars
                    granularity=granularity,
                ),
            )
        # if a resource is added
        elif resource in running_calendars and resource not in reference_calendars:
            # resource is no longer available
            causes.append(
                AnyNode(
                    # what changed? the new resource "resource" is now available
                    what=f"resource '{resource}' is now available",
                    # how changed? include the reference availability calendar for the resource
                    how=Pair(
                        reference=None,
                        running=running_calendars[resource],
                        unit="calendar",
                    ),
                    # data contains the data used to discover the calendars
                    data=Pair(
                        reference=drift.reference_model,
                        running=drift.running_model,
                        unit="events",
                    ),
                    # include the granularity used for computing the calendars
                    granularity=granularity,
                ),
            )
        # if the resource appears in the reference and the running periods, compare the calendars
        else:
            # compute added and removed availability intervals
            # removed_intervals contains the intervals where the resource was available in reference model but
            # not in running model
            removed_intervals: dict[int, IntervalTree] = {}
            # added_intervals contains the intervals where the resource was available in running model but
            # not in reference model
            added_intervals: dict[int, IntervalTree] = {}
            # for each day in the weekly calendar
            for day in range(7):
                # initialize the interval for that day to the reference availability
                removed_intervals[day] = IntervalTree(reference_calendars[resource][day])
                # remove all intervals present in the running calendar for that day
                for interval in running_calendars[resource][day]:
                    removed_intervals[day].chop(interval.begin, interval.end)
                # remove the entry for the day if it is empty
                if len(removed_intervals[day]) == 0:
                    del removed_intervals[day]

                # initialize the interval for the day to the running availability
                added_intervals[day] = IntervalTree(running_calendars[resource][day])
                # remove the intervals where the resource was available in the reference model
                for interval in reference_calendars[resource][day]:
                    added_intervals[day].chop(interval.begin, interval.end)
                # remove the entry for the day if it is empty
                if len(added_intervals[day]) == 0:
                    del added_intervals[day]

            removed = AnyNode(
                # what changed? availability intervals have been removed for a resource
                what=f"removed availability slot for resource '{resource}'",
                # how did it change? store the removed intervals
                how=removed_intervals,
                # data contains the calendars used to compute the difference
                data=Pair(
                    reference=reference_calendars[resource],
                    running=running_calendars[resource],
                    unit="calendar",
                ),
            ) if len(removed_intervals) > 0 else None

            added = AnyNode(
                # what changed? availability intervals have been added for a resource
                what=f"added availability slot for resource '{resource}'",
                # how did it change? store the added intervals
                how=added_intervals,
                # data contains the calendars used to compute the difference
                data=Pair(
                    reference=reference_calendars[resource],
                    running=running_calendars[resource],
                    unit="calendar",
                ),
            ) if len(added_intervals) > 0 else None

            # add a cause if there is any difference in the availability calendars for the resource
            if removed is not None or added is not None:
                causes.append(
                    AnyNode(
                        # what changed? the availability for a resource that appears both before and after the change
                        what=f"availability slots for resource '{resource}' changed",
                        # how did it change? include the calendars for before and after the change
                        how=Pair(
                            reference=reference_calendars[resource],
                            running=running_calendars[resource],
                            unit="calendar",
                        ),
                        # data contains the raw data used to discover the calendars
                        data=Pair(
                            reference=drift.reference_model,
                            running=drift.running_model,
                            unit="events",
                        ),
                        # include the granularity used for computing the calendars
                        granularity=granularity,
                        # the causes of the drift includes the added and removed availability intervals
                        children=[cause for cause in [added, removed] if cause is not None],
                    ),
                )

    # return the list of diferences in the resources availability calendars
    return causes


def __check_extraneous_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time due to extraneous factors changed at a case-level
    if continuous_test(drift.case_features.extraneous_time.reference, drift.case_features.extraneous_time.running):
        # a change in the extraneous times could be explained by changes in features that reflect the extraneous behaviour of the user
        extraneous_factors = __check_attributes(drift,
                                                class_extractor=lambda event: event.waiting_time.extraneous.duration)

        # return the tree explaining the changes
        return AnyNode(
            # what changed? the waiting time due to extraneous factors
            what="case waiting time distribution due to extraneous factors changed!",
            # how did it change? include the distributions for before and after the change
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.extraneous_time.reference),
                running=scipy.stats.describe(drift.case_features.extraneous_time.running),
                unit=drift.case_features.extraneous_time.unit,
            ),
            # data contains the full data used in the test
            data=Pair(
                reference=drift.case_features.extraneous_time.reference,
                running=drift.case_features.extraneous_time.running,
                unit=drift.case_features.extraneous_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the waiting time distribution due to extraneous factors for activity "activity"
                    what=f"activity '{activity}' waiting time distribution due to extraneous factors changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.extraneous_time.reference[activity])
                        if len(list(drift.activity_features.extraneous_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.extraneous_time.running[activity])
                        if len(list(drift.activity_features.extraneous_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.extraneous_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.extraneous_time.reference[activity],
                        running=drift.activity_features.extraneous_time.running[activity],
                        unit=drift.activity_features.extraneous_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if continuous_test(drift.activity_features.extraneous_time.reference[activity],
                                                           drift.activity_features.extraneous_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=[cause for cause in extraneous_factors if cause is not None],
        )
    # if no change is detected, return None
    return None


def __check_batching_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time due to batching changed at a case-level
    if continuous_test(drift.case_features.batching_time.reference, drift.case_features.batching_time.running):
        # changes in the batching waiting time can be explained by changes in the batching policies (both creation and
        # firing) or by changes in the arrival rate
        batching_policies = __check_batching_policies(drift)
        inter_case_time = __check_inter_case_time(drift)
        # if there is a change, return a new tree with the change and its causes
        return AnyNode(
            # what changed? the distribution of waiting times due to batching
            what="case waiting time distribution due to batching changed!",
            # how did it change? include the distribution of times before and after the change
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.batching_time.reference),
                running=scipy.stats.describe(drift.case_features.batching_time.running),
                unit=drift.case_features.batching_time.unit,
            ),
            # data contains the full data used in the test
            data=Pair(
                reference=drift.case_features.batching_time.reference,
                running=drift.case_features.batching_time.running,
                unit=drift.case_features.batching_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the waiting time distribution due to batching for activity "activity"
                    what=f"activity '{activity}' waiting time distribution due to batching changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.batching_time.reference[activity])
                        if len(list(drift.activity_features.batching_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.batching_time.running[activity])
                        if len(list(drift.activity_features.batching_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.batching_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.batching_time.reference[activity],
                        running=drift.activity_features.batching_time.running[activity],
                        unit=drift.activity_features.batching_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if continuous_test(drift.activity_features.batching_time.reference[activity],
                                                           drift.activity_features.batching_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=[cause for cause in [inter_case_time, *batching_policies] if cause is not None],
        )
    # if no change is found, return None
    return None


def __check_prioritization_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time due to prioritization changed at a case-level
    if continuous_test(drift.case_features.prioritization_time.reference, drift.case_features.prioritization_time.running):
        # if a change in the prioritization times is found, it may be due to changes in the arrival rate (or the inter
        # case time), in the case length, in the weekly available hours (the "capacity" of the system) or in the
        # prioritization rules
        inter_case_times = __check_inter_case_time(drift)
        case_length = __check_case_length(drift)
        weekly_available_hours = __check_weekly_available_hours(drift)
        priorities = __check_prioritization_policies(drift)
        # return the subtree with the change
        return AnyNode(
            # what changed? the distribution of waiting times due to prioritization
            what="case waiting time distribution due to prioritization changed!",
            # how did it change? include the distributions of time for both before and after the drift
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.prioritization_time.reference),
                running=scipy.stats.describe(drift.case_features.prioritization_time.running),
                unit=drift.case_features.prioritization_time.unit,
            ),
            # data contains the raw data used to perform the test
            data=Pair(
                reference=drift.case_features.prioritization_time.reference,
                running=drift.case_features.prioritization_time.running,
                unit=drift.case_features.prioritization_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the waiting time distribution due to prioritization for activity "activity"
                    what=f"activity '{activity}' waiting time distribution due to prioritization changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.prioritization_time.reference[activity])
                        if len(list(drift.activity_features.prioritization_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.prioritization_time.running[activity])
                        if len(list(drift.activity_features.prioritization_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.prioritization_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.prioritization_time.reference[activity],
                        running=drift.activity_features.prioritization_time.running[activity],
                        unit=drift.activity_features.prioritization_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if
                continuous_test(drift.activity_features.prioritization_time.reference[activity],
                     drift.activity_features.prioritization_time.running[activity])
            ],
            # include the causes of the drift as children ot the tree
            children=[cause for cause in [inter_case_times, case_length, weekly_available_hours, *priorities] if
                      cause is not None],

        )

    # if no drift found, return None
    return None


def __check_contention_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time due to contention changed at a case-level
    if continuous_test(drift.case_features.contention_time.reference, drift.case_features.contention_time.running):
        # if there is a change in the contention time, maybe it comes from a change in the arrival rate (in this case
        # computed as the inter case time), the case length or the weekly available resource hours (i.e., the weekly "capacity").
        inter_case_time = __check_inter_case_time(drift)
        case_length = __check_case_length(drift)
        weekly_available_hours = __check_weekly_available_hours(drift)

        # return a subtree with the change and its causes
        return AnyNode(
            # what changed? the case waiting time distribution due to contention
            what="case waiting time distribution due to contention changed!",
            # how did they change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.contention_time.reference),
                running=scipy.stats.describe(drift.case_features.contention_time.running),
                unit=drift.case_features.contention_time.unit,
            ),
            # data contains the full data used in the test
            data=Pair(
                reference=drift.case_features.contention_time.reference,
                running=drift.case_features.contention_time.running,
                unit=drift.case_features.contention_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the waiting time distribution due to contention for activity "activity"
                    what=f"activity '{activity}' waiting time distribution due to contention changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.contention_time.reference[activity])
                        if len(list(drift.activity_features.contention_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.contention_time.running[activity])
                        if len(list(drift.activity_features.contention_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.contention_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.contention_time.reference[activity],
                        running=drift.activity_features.contention_time.running[activity],
                        unit=drift.activity_features.contention_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if continuous_test(drift.activity_features.contention_time.reference[activity],
                                                           drift.activity_features.contention_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=[cause for cause in [inter_case_time, case_length, weekly_available_hours] if cause is not None],
        )

    return None


def __check_unavailability_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time due to resources unavailability changed at a case-level
    if continuous_test(drift.case_features.availability_time.reference, drift.case_features.availability_time.running):
        # if the time due to resource unavailability changed, maybe the resources availability calendars changed too
        resources_availability = __check_resources_availability(drift)

        return AnyNode(
            # what changed? the case waiting time due to resources unavailability
            what="case waiting time distribution due to resource unavailability changed!",
            # how did they change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.availability_time.reference),
                running=scipy.stats.describe(drift.case_features.availability_time.running),
                unit=drift.case_features.availability_time.unit,
            ),
            # data contains the full data used in the test
            data=Pair(
                reference=drift.case_features.availability_time.reference,
                running=drift.case_features.availability_time.running,
                unit=drift.case_features.availability_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the waiting time distribution due to resource unavailability for activity "activity"
                    what=f"activity '{activity}' waiting time distribution due to resource unavailability changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.availability_time.reference[activity])
                        if len(list(drift.activity_features.availability_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.availability_time.running[activity])
                        if len(list(drift.activity_features.availability_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.availability_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.availability_time.reference[activity],
                        running=drift.activity_features.availability_time.running[activity],
                        unit=drift.activity_features.availability_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if
                continuous_test(drift.activity_features.availability_time.reference[activity],
                     drift.activity_features.availability_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=resources_availability,
        )

    return None


def __check_waiting_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time changed at a pre-case level
    if continuous_test(drift.case_features.waiting_time.reference, drift.case_features.waiting_time.running):
        # the causes for a change in the waiting time can be decomposed in changes in the resources availability time,
        # changes in the contention time, changes in the prioritization time, changes in the batching times and changes
        # in the extraneous times
        unavailability = __check_unavailability_times(drift)
        contention = __check_contention_times(drift)
        prioritization = __check_prioritization_times(drift)
        batching = __check_batching_times(drift)
        extraneous = __check_extraneous_times(drift)

        # create a tree with the change
        return AnyNode(
            # what changed? the case waiting time distribution
            what="case waiting time distribution changed!",
            # how did they change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.waiting_time.reference),
                running=scipy.stats.describe(drift.case_features.waiting_time.running),
                unit=drift.case_features.waiting_time.unit,
            ),
            # data contains the full data used in the test
            data=Pair(
                reference=drift.case_features.waiting_time.reference,
                running=drift.case_features.waiting_time.running,
                unit=drift.case_features.waiting_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the waiting time distribution for activity "activity"
                    what=f"activity '{activity}' waiting time distribution changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.waiting_time.reference[activity])
                        if len(list(drift.activity_features.waiting_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.waiting_time.running[activity])
                        if len(list(drift.activity_features.waiting_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.waiting_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.waiting_time.reference[activity],
                        running=drift.activity_features.waiting_time.running[activity],
                        unit=drift.activity_features.waiting_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if continuous_test(drift.activity_features.waiting_time.reference[activity],
                                                           drift.activity_features.waiting_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=[cause for cause in [unavailability, contention, prioritization, batching, extraneous] if
                      cause is not None],
        )

    # if no change is found, return None
    return None


def __check_effective_times(drift: Drift) -> AnyNode | None:
    # check if the effective execution time changed at a case-level
    if continuous_test(drift.case_features.effective_time.reference, drift.case_features.effective_time.running):
        # if there is a change, maybe the cases are more complex, or maybe they require doing more tasks to finish them
        case_complexity = __check_attributes(drift,
                                             class_extractor=lambda event: event.processing_time.effective.duration)
        case_length = __check_case_length(drift)

        # add a node to the tree reporting the change in the effective processing time
        return AnyNode(
            # what changed? the effective time needed to finish a case execution
            what="case effective processing time distribution changed!",
            # how did it change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe(drift.case_features.effective_time.reference),
                running=scipy.stats.describe(drift.case_features.effective_time.running),
                unit=drift.case_features.effective_time.unit,
            ),
            # data contains the full data used to perform the test
            data=Pair(
                reference=drift.case_features.effective_time.reference,
                running=drift.case_features.effective_time.running,
                unit=drift.case_features.effective_time.unit,
            ),
            # store the changes per activity
            changes_per_activity=[
                AnyNode(
                    # what changed? the effective time distribution for activity "activity"
                    what=f"activity '{activity}' idle processing time distribution changed!",
                    # how did it change? include the distributions for both pre- and post- drift data
                    how=Pair(
                        # if no values present for the activity, return None instead of the distribution description
                        reference=scipy.stats.describe(drift.activity_features.effective_time.reference[activity])
                        if len(list(drift.activity_features.effective_time.reference[activity])) > 0 else None,
                        # if no values present for the activity, return None instead of the distribution description
                        running=scipy.stats.describe(drift.activity_features.effective_time.running[activity])
                        if len(list(drift.activity_features.effective_time.running[activity])) > 0 else None,
                        unit=drift.activity_features.effective_time.unit,
                    ),
                    # data contains the raw data used in the test
                    data=Pair(
                        reference=drift.activity_features.effective_time.reference[activity],
                        running=drift.activity_features.effective_time.running[activity],
                        unit=drift.activity_features.effective_time.unit,
                    ),
                    # check every activity in the sublogs
                ) for activity in drift.activities if continuous_test(drift.activity_features.effective_time.reference[activity],
                                                           drift.activity_features.effective_time.running[activity])
            ],
            # the causes of this change can be the changes in the case length or in the cases complexity
            children=[cause for cause in [*case_complexity, case_length] if cause is not None],
        )
    # if no change is detected in the effective time, return None
    return None


def __check_idle_times(drift: Drift) -> AnyNode | None:
    # check if the idle execution time changed at a per-case level
    if continuous_test(
            [event.processing_time.idle.duration.total_seconds() for event in drift.reference_model],
            [event.processing_time.idle.duration.total_seconds() for event in drift.running_model],
    ):
        # if the idle time changed, maybe the resources availability changed
        resources_availability = __check_resources_availability(drift)

        return AnyNode(
            # what changed? the idle processing times
            what="idle processing time distribution changed!",
            # how did it change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe([event.processing_time.idle.duration.total_seconds() for event in drift.reference_model]),
                running=scipy.stats.describe([event.processing_time.idle.duration.total_seconds() for event in drift.running_model]),
            ),
            # data contains the full data used for evaluating the change
            data=Pair(
                reference=[event.processing_time.idle.duration.total_seconds() for event in drift.reference_model],
                running=[event.processing_time.idle.duration.total_seconds() for event in drift.running_model],
            ),
            # the causes for the change are the changes in the resources availability
            children=resources_availability if resources_availability is not None else [],
        )
    # return None if no changes found in idle processing time
    return None


def __check_processing_times(drift: Drift) -> AnyNode | None:
    # check if the processing time changed
    if continuous_test(
            [event.processing_time.total.duration.total_seconds() for event in drift.reference_model],
            [event.processing_time.total.duration.total_seconds() for event in drift.running_model],
    ):
        # if a drift is detected in the processing time, check for changes in the effective and idle times
        effective = __check_effective_times(drift)
        idle = __check_idle_times(drift)

        # add a node to the tree reporting the change in the processing times
        return AnyNode(
            # what changed? the processing time
            what="processing time distribution changed!",
            # how did it change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe([event.processing_time.total.duration.total_seconds() for event in drift.reference_model]),
                running=scipy.stats.describe([event.processing_time.total.duration.total_seconds() for event in drift.running_model]),
                unit="seconds",
            ),
            # data contains the raw data used in the test
            data=Pair(
                reference=[event.processing_time.total.duration.total_seconds() for event in drift.reference_model],
                running=[event.processing_time.total.duration.total_seconds() for event in drift.running_model],
                unit="seconds",
            ),
            # the causes of the drift are the changes in the effective and the idle processing times
            children=[cause for cause in [effective, idle] if cause is not None],
        )
    # return None if no changes found in processing time
    return None


@profile("drift explanation")
def explain_drift(drift: Drift) -> AnyNode | None:
    """Build a tree with the causes that explain the drift characterized by the given drift features"""
    # if there is a drift in the cycle time distribution, check for drifts in the waiting and processing times and build
    # a tree accordingly, explaining the changes that occurred to the process
    if continuous_test(
            [event.cycle_time.total_seconds() for event in drift.reference_model],
            [event.cycle_time.total_seconds() for event in drift.running_model],
    ):
        # check waiting and processing times for changes
        waiting = __check_waiting_times(drift)
        processing = __check_processing_times(drift)

        # create a tree with the change
        return AnyNode(
            # what changed? the cycle time distribution
            what="cycle time distribution changed!",
            # how did it change? include the distributions for both pre- and post- drift data
            how=Pair(
                reference=scipy.stats.describe([event.cycle_time.total_seconds() for event in drift.reference_model]),
                running=scipy.stats.describe([event.cycle_time.total_seconds() for event in drift.running_model]),
            ),
            # data contains the full data used for evaluating the change
            data=Pair(
                reference=[event.cycle_time.total_seconds() for event in drift.reference_model],
                running=[event.cycle_time.total_seconds() for event in drift.running_model],
            ),
            # what are the causes of the drift? the subtrees resulting from checking the waiting and processing times
            # they are added as children so the tree has a hierarchy
            children=[cause for cause in [waiting, processing] if cause is not None],
        )

    # if no change is found between both running and reference data, return None
    return None
