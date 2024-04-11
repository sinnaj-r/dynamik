from __future__ import annotations

import itertools
import typing
from datetime import timedelta

import pandas as pd
import scipy
from anytree import AnyNode
from pandas import CategoricalDtype

from expert.drift.model import Drift, DriftCause, Pair
from expert.logger import LOGGER
from expert.model import Event
from expert.timer import profile
from expert.utils.batching import (
    build_batch_creation_features,
    build_batch_firing_features,
    discover_batch_creation_policies,
    discover_batch_firing_policies,
)
from expert.utils.calendars import Calendar, compute_weekly_available_time_per_resource, discover_calendars2
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
        causes.append(__check_policy(creation_policy, reference_creation_features, running_creation_features,
                                     policy_type="batch creation"))

        # after evaluating the creation policy, evaluate its firing policies
        # filter events in the current batch
        events_in_batch = filter_log(creation_policy, drift.reference_model)
        # discover the batch firing rules for the given creation policy
        firing_policies = discover_batch_firing_policies(events_in_batch)

        # check every reference firing rule score with running and reference models
        for firing_policy in firing_policies:
            causes.append(__check_policy(firing_policy, reference_firing_features, running_firing_features,
                                         policy_type=f"batch '{creation_policy.__repr__()}' firing"))

    # check running creation policies against reference and running
    for creation_policy in running_creation_policies:
        # check the batching creation policy
        causes.append(__check_policy(creation_policy, reference_creation_features, running_creation_features,
                                     policy_type="batch creation"))

        # after evaluating the creation policy, evaluate its firing policies
        # filter events in the current batch
        events_in_batch = filter_log(creation_policy, drift.running_model)
        # discover the batch firing rules for the given creation policy
        firing_policies = discover_batch_firing_policies(events_in_batch)

        # check every firing rule score with running and reference models
        for firing_policy in firing_policies:
            causes.append(__check_policy(firing_policy, reference_firing_features, running_firing_features,
                                         policy_type=f"batch '{creation_policy.__repr__()}' firing"))
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
                ) for activity in drift.activities if
                continuous_test(drift.activity_features.extraneous_time.reference[activity],
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
                ) for activity in drift.activities if
                continuous_test(drift.activity_features.batching_time.reference[activity],
                                drift.activity_features.batching_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=[cause for cause in [inter_case_time, *batching_policies] if cause is not None],
        )
    # if no change is found, return None
    return None


def __check_prioritization_times(drift: Drift) -> AnyNode | None:
    # check if the waiting time due to prioritization changed at a case-level
    if continuous_test(drift.case_features.prioritization_time.reference,
                       drift.case_features.prioritization_time.running):
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
                ) for activity in drift.activities if
                continuous_test(drift.activity_features.contention_time.reference[activity],
                                drift.activity_features.contention_time.running[activity])
            ],
            # the causes of the drift in the waiting time can be decomposed in the waiting time canvas components
            children=[cause for cause in [inter_case_time, case_length, weekly_available_hours] if cause is not None],
        )

    return None


class DriftExplainer:
    """TODO docs"""

    drift: Drift

    def __init__(
            self: typing.Self,
            drift: Drift,
    ) -> None:
        self.drift = drift

    def __describe_distributions(self: typing.Self, extractor: typing.Callable[[Event], timedelta]) -> Pair:
        return Pair(
            reference=scipy.stats.describe(
                [extractor(event).total_seconds() for event in self.drift.reference_model.data]),
            running=scipy.stats.describe([extractor(event).total_seconds() for event in self.drift.running_model.data]),
        )

    def __describe_calendars(self: typing.Self) -> Pair:
        reference_calendars = discover_calendars2(self.drift.reference_model.data)
        running_calendars = discover_calendars2(self.drift.running_model.data)

        return Pair(
            reference=sum(
                [calendar.transform(lambda value: min(value, 1)) for calendar in reference_calendars.values()],
                Calendar(),
            ),
            running=sum(
                [calendar.transform(lambda value: min(value, 1)) for calendar in running_calendars.values()],
                Calendar(),
            ),
        )

    def __get_calendars(self: typing.Self) -> Pair:
        return Pair(
            reference=discover_calendars2(self.drift.reference_model.data),
            running=discover_calendars2(self.drift.running_model.data),
        )

    def __get_data(self: typing.Self, extractor: typing.Callable[[Event], timedelta]) -> Pair:
        """TODO docs"""
        return Pair(
            reference=[extractor(event) for event in self.drift.reference_model.data],
            running=[extractor(event) for event in self.drift.running_model.data],
        )

    def drift_in_time(self: typing.Self, extractor: typing.Callable[[Event], timedelta], *, significance: float = 0.05) -> bool:
        """TODO docs"""
        if len(self.drift.reference_model.data) > 0 and len(self.drift.running_model.data) > 0:
            result = scipy.stats.kstest(
                [extractor(event).total_seconds() for event in self.drift.reference_model.data],
                [extractor(event).total_seconds() for event in self.drift.running_model.data],
            )

            LOGGER.verbose("test(reference != running) p-value: %.4f", result.pvalue)

            return result.pvalue < significance

        return False

    def drift_in_availability(self: typing.Self) -> bool:
        """TODO docs"""
        # discover calendars for reference and running models
        reference_calendars = discover_calendars2(self.drift.reference_model.data)
        running_calendars = discover_calendars2(self.drift.running_model.data)
        # aggregate calendars by resource, so each slot contains the number of resources available
        aggregated_reference_calendar = sum(
            [calendar.transform(lambda value: min(value, 1)) for calendar in reference_calendars.values()],
            Calendar(),
        )
        aggregated_running_calendar = sum(
            [calendar.transform(lambda value: min(value, 1)) for calendar in running_calendars.values()],
            Calendar(),
        )
        # compare aggregated calendars
        results = []
        for slot in aggregated_reference_calendar.slots:
            results.append(aggregated_reference_calendar[slot] != aggregated_running_calendar[slot])
            LOGGER.verbose(
                "test(reference calendar[%s] != running calendar[%s]) result: %s",
                slot, slot, results[-1],
            )

        return any(results)

    def build_time_descriptor(
            self: typing.Self,
            title: str,
            extractor: typing.Callable[[Event], timedelta],
            parent: DriftCause | None = None,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? the processing time
            what=title,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_distributions(extractor),
            # data contains the raw data used in the test
            data=self.__get_data(extractor),
            # the causes of the drift are the changes in the effective and the idle processing times
            parent=parent,
        )

    def build_calendar_descriptor(
            self: typing.Self,
            title: str,
            parent: DriftCause | None = None,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? the processing time
            what=title,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_calendars(),
            # data contains the raw data used in the test
            data=self.__get_calendars(),
            # the causes of the drift are the changes in the effective and the idle processing times
            parent=parent,
        )


@profile()
def explain_drift(drift: Drift) -> AnyNode | None:
    """Build a tree with the causes that explain the drift characterized by the given drift features"""
    # if there is a drift in the cycle time distribution, check for drifts in the waiting and processing times and build
    # a tree accordingly, explaining the changes that occurred to the process
    explainer = DriftExplainer(drift)
    root_cause = explainer.build_time_descriptor(
        "cycle time distribution changed!",
        lambda event: event.cycle_time,
    )

    # check processing time
    if explainer.drift_in_time(lambda event: event.processing_time.total.duration):
        # add a node to the tree reporting the change in the processing times
        processing_time = explainer.build_time_descriptor(
            "processing time distribution changed!",
            lambda event: event.processing_time.total.duration,
            parent=root_cause,
        )

        # check effective processing time
        if explainer.drift_in_time(lambda event: event.processing_time.effective.duration):
            effective_time = explainer.build_time_descriptor(
                "effective processing time distribution changed!",
                lambda event: event.processing_time.effective.duration,
                parent=processing_time,
            )
            # TODO analyze causes

        # check idle processing time
        if explainer.drift_in_time(lambda event: event.processing_time.idle.duration):
            idle_time = explainer.build_time_descriptor(
                "idle processing time distribution changed!",
                lambda event: event.processing_time.idle.duration,
                parent=processing_time,
            )

            # check changes in the availability calendars
            if explainer.drift_in_availability():
                explainer.build_calendar_descriptor(
                    "resource availability calendars changed!",
                    parent=idle_time,
                )

    # check waiting time
    if explainer.drift_in_time(lambda event: event.waiting_time.total.duration):
        # add a node to the tree reporting the change in the waiting times
        waiting_time = explainer.build_time_descriptor(
            "waiting time distribution changed!",
            lambda event: event.waiting_time.total.duration,
            parent=root_cause,
        )

        # check waiting time due to unavailability
        if explainer.drift_in_time(lambda event: event.waiting_time.availability.duration):
            unavailability_time = explainer.build_time_descriptor(
                "waiting time due to resource unavailability distribution changed!",
                lambda event: event.waiting_time.availability.duration,
                parent=waiting_time,
            )

            # check changes in the availability calendars
            if explainer.drift_in_availability():
                explainer.build_calendar_descriptor(
                    "resource availability calendars changed!",
                    parent=unavailability_time,
                )

        # check waiting time due to contention
        if explainer.drift_in_time(lambda event: event.waiting_time.contention.duration):
            contention_time = explainer.build_time_descriptor(
                "waiting time due to resource contention distribution changed!",
                lambda event: event.waiting_time.contention.duration,
                parent=waiting_time,
            )

        # check waiting time due to prioritization
        if explainer.drift_in_time(lambda event: event.waiting_time.prioritization.duration):
            prioritization_time = explainer.build_time_descriptor(
                "waiting time due to prioritization distribution changed!",
                lambda event: event.waiting_time.prioritization.duration,
                parent=waiting_time,
            )

        # check waiting time due to batching
        if explainer.drift_in_time(lambda event: event.waiting_time.batching.duration):
            batching_time = explainer.build_time_descriptor(
                "waiting time due to batching distribution changed!",
                lambda event: event.waiting_time.batching.duration,
                parent=waiting_time,
            )

        # check waiting time due to extraneous
        if explainer.drift_in_time(lambda event: event.waiting_time.extraneous.duration):
            extraneous_time = explainer.build_time_descriptor(
                "waiting time due to extraneous distribution changed!",
                lambda event: event.waiting_time.extraneous.duration,
                parent=waiting_time,
            )

    # create a tree with the change
    return root_cause
