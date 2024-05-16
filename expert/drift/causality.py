from __future__ import annotations

import itertools
import typing
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median

import pandas as pd
import scipy

from expert.drift.model import Drift, DriftCause
from expert.model import Event, Log, Resource
from expert.utils.logger import LOGGER
from expert.utils.model import DistributionDescription, HashableDF, Pair
from expert.utils.pm.batching import build_batch_creation_features, build_batch_firing_features
from expert.utils.pm.calendars import Calendar, discover_calendars
from expert.utils.pm.prioritization import build_prioritization_features
from expert.utils.pm.profiles import ActivityProfile, Profile, ResourceProfile
from expert.utils.rules import ConfusionMatrix, Rule, compute_rule_score, discover_rules, filter_log


class DriftExplainer:
    """TODO docs"""

    drift: Drift
    significance: float
    calendar_threshold: int

    def __init__(self: typing.Self, drift: Drift, significance: float) -> None:
        self.drift = drift
        self.significance = significance
        self.calendar_threshold = calendar_threshold

    def __describe_distributions(
            self: typing.Self,
            extractor: typing.Callable[[Event], timedelta],
    ) -> Pair[DistributionDescription]:
        return Pair(
            reference=DistributionDescription(
                scipy.stats.describe(
                    [extractor(event).total_seconds() for event in self.drift.reference_model.data],
                ),
            ),
            running=DistributionDescription(
                scipy.stats.describe(
                    [extractor(event).total_seconds() for event in self.drift.running_model.data],
                ),
            ),
        )

    def __describe_calendars(
            self: typing.Self,
    ) -> Pair[Calendar]:
        reference_calendars = discover_calendars(self.drift.reference_model.data)
        running_calendars = discover_calendars(self.drift.running_model.data)

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

    def __describe_rates(
            self: typing.Self,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> Pair[Calendar]:
        data = self.__get_rate_data(filter_, extractor)

        return Pair(
            reference=data.reference - data.running,
            running=data.running - data.reference,
        )

    def __describe_policies(
            self: typing.Self,
            feature_extractor: typing.Callable[[Log], pd.DataFrame],
            filter_: typing.Callable[[Log], Log] = lambda _: _,
    ) -> Pair[typing.Mapping[Rule, ConfusionMatrix]]:
        # extract features for pre- and post-drift
        reference_features = feature_extractor(filter_(self.drift.reference_model.data))
        running_features = feature_extractor(filter_(self.drift.running_model.data))
        # discover rules for pre- and post-drift
        reference_policies = discover_rules(HashableDF(reference_features))
        running_policies = discover_rules(HashableDF(running_features))

        results = Pair(reference={}, running={})
        # evaluate each rule and add the score to the results pair
        for rule in itertools.chain(reference_policies, running_policies):
            results.reference[rule] = compute_rule_score(rule, HashableDF(reference_features))
            results.running[rule] = compute_rule_score(rule, HashableDF(running_features))

        return results

    def __describe_profiles(
            self: typing.Self,
            profile_builder: typing.Callable[[Log], Profile],
    ) -> Pair[Profile]:
        reference_profile = profile_builder(self.drift.reference_model.data)
        running_profile = profile_builder(self.drift.running_model.data)

        return Pair(
            reference=reference_profile,
            running=running_profile,
        )

    def __get_calendars(self: typing.Self) -> Pair[typing.Mapping[Resource, Calendar]]:
        return Pair(
            reference=discover_calendars(self.drift.reference_model.data),
            running=discover_calendars(self.drift.running_model.data),
        )

    def __get_time_data(
            self: typing.Self,
            time_extractor: typing.Callable[[Event], timedelta],
    ) -> Pair[typing.Iterable[timedelta]]:
        """TODO docs"""
        return Pair(
            reference=[time_extractor(event) for event in self.drift.reference_model.data],
            running=[time_extractor(event) for event in self.drift.running_model.data],
        )

    def __get_rate_data(
            self: typing.Self,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> Pair[Calendar]:
        return Pair(
            reference=Calendar.discover(
                [event for event in self.drift.reference_model.data if filter_(event)],
                lambda event: [extractor(event)],
            ),
            running=Calendar.discover(
                [event for event in self.drift.running_model.data if filter_(event)],
                lambda event: [extractor(event)],
            ),
        )

    def __get_policies_data(
            self: typing.Self,
            filter_: typing.Callable[[Log], Log] = lambda _: _,
    ) -> Pair[Log]:
        return Pair(
            reference=filter_(self.drift.reference_model.data),
            running=filter_(self.drift.running_model.data),
        )

    def __get_profiles(
            self: typing.Self,
            profile_builder: typing.Callable[[Log], Profile],
    ) -> Pair[Profile]:
        return Pair(
            reference=profile_builder(self.drift.reference_model.data),
            running=profile_builder(self.drift.running_model.data),
        )

    def has_drift_in_time(
            self: typing.Self,
            time_extractor: typing.Callable[[Event], timedelta],
    ) -> bool:
        """TODO docs"""
        reference_events_per_activity = defaultdict(list)
        for event in self.drift.reference_model.data:
            reference_events_per_activity[event.activity].append(event.cycle_time)

        reference_times_per_activity = defaultdict(timedelta)
        for activity, values in reference_events_per_activity.items():
            reference_times_per_activity[activity] = median(values)

        if len(self.drift.reference_model.data) > 0 and len(self.drift.running_model.data) > 0:
            result = scipy.stats.kstest(
                [time_extractor(event) / reference_times_per_activity[event.activity] for event in self.drift.reference_model.data],
                [time_extractor(event) / reference_times_per_activity[event.activity] for event in self.drift.running_model.data],
            )

            LOGGER.verbose("test(reference != running) p-value: %.4f", result.pvalue)

            return result.pvalue < self.significance

        return False

    def has_drift_in_calendar(
            self: typing.Self,
    ) -> bool:
        """TODO docs"""
        # discover calendars for reference and running models
        reference_calendars = discover_calendars(self.drift.reference_model.data)
        running_calendars = discover_calendars(self.drift.running_model.data)
        # aggregate calendars by resource, so each slot contains the number of resources available
        aggregated_reference_calendar = sum(
            [calendar.transform(lambda value: min(value, 1)) for calendar in reference_calendars.values()],
            Calendar(),
        )
        aggregated_running_calendar = sum(
            [calendar.transform(lambda value: min(value, 1)) for calendar in running_calendars.values()],
            Calendar(),
        )
        # compute the difference between calendars, saving the absolute value of the difference
        diff = (aggregated_reference_calendar - aggregated_running_calendar).transform(lambda value: abs(value))

        # check if diff is greater than threshold
        return sum([value for _, value in diff]) >= self.calendar_threshold

    def has_drift_in_rate(
            self: typing.Self,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> bool:
        """TODO docs"""
        # compute the average reference rate distribution per hour and day of week
        reference_arrival_rates = Calendar.discover(
            [event for event in self.drift.reference_model.data if filter_(event)],
            lambda event: [extractor(event)],
        )

        # compute the average running arrival distribution per hour and day of week
        running_arrival_rates = Calendar.discover(
            [event for event in self.drift.running_model.data if filter_(event)],
            lambda event: [extractor(event)],
        )

        reference_size = sum(reference_arrival_rates[slot] for slot in reference_arrival_rates.slots)
        running_size = sum(running_arrival_rates[slot] for slot in running_arrival_rates.slots)
        results = {}

        if reference_size == 0 or running_size == 0:
            LOGGER.warning("can not check rate in models")
            LOGGER.warning("no cases start or end in this window")
            LOGGER.warning("try increasing the window size")
            return False

        for key in reference_arrival_rates.slots:
            # we use the combined size for assessing differences in the total also
            # (otherwise if the changes are proportional to the sample size nothing will be detected)
            results[key] = scipy.stats.poisson_means_test(
                reference_arrival_rates[key],
                reference_size + running_size,
                running_arrival_rates[key],
                reference_size + running_size,
            )
            LOGGER.verbose("test(reference(%s) != running(%s)) p-value: %.4f", key, key, results[key].pvalue)

        return scipy.stats.combine_pvalues([
            value.pvalue for value in results.values()
        ]).pvalue < self.significance

    def has_drift_in_policies(
            self: typing.Self,
            feature_extractor: typing.Callable[[Log], pd.DataFrame],
            filter_: typing.Callable[[Log], Log] = lambda _: _,
    ) -> bool:
        """TODO docs"""
        reference_features = feature_extractor(filter_(self.drift.reference_model.data))
        running_features = feature_extractor(filter_(self.drift.running_model.data))

        reference_policies = discover_rules(HashableDF(reference_features))
        running_policies = discover_rules(HashableDF(running_features))

        results = []

        for rule in itertools.chain(reference_policies, running_policies):
            reference_scores = compute_rule_score(rule, HashableDF(reference_features), n_samples=20, sample_size=0.8)
            running_scores = compute_rule_score(rule, HashableDF(running_features), n_samples=20, sample_size=0.8)

            results.append(
                scipy.stats.kstest(
                    [score.f1_score for score in reference_scores],
                    [score.f1_score for score in running_scores],
                ),
            )
            LOGGER.verbose("test(reference != running) p-value: %.4f", results[-1].pvalue)

        return scipy.stats.combine_pvalues([
            result.pvalue for result in results
        ]).pvalue < self.significance

    def has_drift_in_profile(
            self: typing.Self,
            profile_builder: typing.Callable[[Log], Profile],
    ) -> bool:
        """TODO docs"""
        reference_profile = profile_builder(self.drift.reference_model.data)
        running_profile = profile_builder(self.drift.running_model.data)

        return reference_profile.statistically_equals(running_profile).pvalue < self.significance

    def build_time_descriptor(
            self: typing.Self,
            title: str,
            time_extractor: typing.Callable[[Event], timedelta],
            parent: DriftCause | None = None,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? the processing time
            what=title,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_distributions(time_extractor),
            # data contains the raw data used in the test
            data=self.__get_time_data(time_extractor),
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

    def build_rate_descriptor(
            self: typing.Self,
            title: str,
            parent: DriftCause | None = None,
            *,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? the processing time
            what=title,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_rates(filter_, extractor),
            # data contains the raw data used in the test
            data=self.__get_rate_data(filter_, extractor),
            # the causes of the drift are the changes in the effective and the idle processing times
            parent=parent,
        )

    def build_policies_descriptor(
            self: typing.Self,
            title: str,
            parent: DriftCause | None = None,
            *,
            feature_extractor: typing.Callable[[Log], pd.DataFrame],
            filter_: typing.Callable[[Log], Log] = lambda _: _,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? policies
            what=title,
            # how did it change? include the policies for both pre- and post- drift data
            how=self.__describe_policies(feature_extractor, filter_),
            # data contains the evaluation of policies for before and after
            data=self.__get_policies_data(filter_),
            parent=parent,
        )

    def build_profile_descriptor(
            self: typing.Self,
            title: str,
            parent: DriftCause | None = None,
            *,
            profile_builder: typing.Callable[[Log], Profile],
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? policies
            what=title,
            # how did it change? include the policies for both pre- and post- drift data
            how=self.__describe_profiles(profile_builder),
            # data contains the evaluation of policies for before and after
            data=self.__get_profiles(profile_builder),
            parent=parent,
        )


def explain_drift(
        drift: Drift,
        *,
        first_activity: str,
        last_activity: str,
        significance: float = 0.05,
        calendar_threshold: int = 0,
) -> DriftCause:
    """Build a tree with the causes that explain the drift characterized by the given drift features"""
    # if there is a drift in the cycle time distribution, check for drifts in the waiting and processing times and build
    # a tree accordingly, explaining the changes that occurred to the process
    explainer = DriftExplainer(drift, significance, calendar_threshold)
    root_cause = explainer.build_time_descriptor(
        "cycle time changed!",
        lambda event: event.cycle_time,
    )

    # check processing time
    LOGGER.verbose("checking drifts in total processing time")
    if explainer.has_drift_in_time(lambda event: event.processing_time.total.duration):
        # add a node to the tree reporting the change in the processing times
        processing_time = explainer.build_time_descriptor(
            "processing time changed!",
            lambda event: event.processing_time.total.duration,
            parent=root_cause,
        )

        # check drift in processing time when the resource is available
        LOGGER.verbose("checking drifts in effective processing time")
        if explainer.has_drift_in_time(lambda event: event.processing_time.effective.duration):
            effective_time = explainer.build_time_descriptor(
                "processing time with resources available changed!",
                lambda event: event.processing_time.effective.duration,
                parent=processing_time,
            )

            # check drifts in the activity profiles
            LOGGER.verbose("checking drifts in activity profiles")
            if explainer.has_drift_in_profile(ActivityProfile.discover):
                explainer.build_profile_descriptor(
                    "activity profiles changed!",
                    parent=effective_time,
                    profile_builder=ActivityProfile.discover,
                )
            # check drifts in the resource profiles
            LOGGER.verbose("checking drifts in resource profiles")
            if explainer.has_drift_in_profile(ResourceProfile.discover):
                explainer.build_profile_descriptor(
                    "resource profiles changed!",
                    parent=effective_time,
                    profile_builder=ResourceProfile.discover,
                )

        # check drift in processing time when the resource is not available
        LOGGER.verbose("checking drifts in idle processing time")
        if explainer.has_drift_in_time(lambda event: event.processing_time.idle.duration):
            idle_time = explainer.build_time_descriptor(
                "processing time with resources unavailable changed!",
                lambda event: event.processing_time.idle.duration,
                parent=processing_time,
            )

            # check changes in the availability calendars
            LOGGER.verbose("checking drifts in calendars")
            if explainer.has_drift_in_calendar():
                explainer.build_calendar_descriptor(
                    "resource availability calendars changed!",
                    parent=idle_time,
                )

    # check waiting time
    LOGGER.verbose("checking drifts in total waiting time")
    if explainer.has_drift_in_time(lambda event: event.waiting_time.total.duration):
        # add a node to the tree reporting the change in the waiting times
        waiting_time = explainer.build_time_descriptor(
            "waiting time changed!",
            lambda event: event.waiting_time.total.duration,
            parent=root_cause,
        )

        # check waiting time due to batching
        LOGGER.verbose("checking drifts in batching waiting time")
        if explainer.has_drift_in_time(lambda event: event.waiting_time.batching.duration):
            batching_time = explainer.build_time_descriptor(
                "waiting time due to batching changed!",
                lambda event: event.waiting_time.batching.duration,
                parent=waiting_time,
            )

            # check changes in the batch creation policies
            LOGGER.verbose("checking drifts batch creation policies")
            if explainer.has_drift_in_policies(build_batch_creation_features):
                explainer.build_policies_descriptor(
                    "batch creation policies changed!",
                    parent=batching_time,
                    feature_extractor=build_batch_creation_features,
                )

            # get the creation policies from the descriptor
            creation_policies = explainer.build_policies_descriptor(
                "",
                feature_extractor=build_batch_creation_features,
            ).how.reference.keys()

            # check changes in the firing policies for each creation policy
            for creation_policy in creation_policies:
                LOGGER.verbose("checking drifts in batch firing policies")
                if explainer.has_drift_in_policies(build_batch_firing_features, filter_log(creation_policy)):
                    explainer.build_policies_descriptor(
                        f"batch firing policies for '{creation_policy}' changed!",
                        parent=batching_time,
                        feature_extractor=build_batch_firing_features,
                        filter_=filter_log(creation_policy),
                    )

        # check waiting time due to contention
        LOGGER.verbose("checking drifts in contention waiting time")
        if explainer.has_drift_in_time(lambda event: event.waiting_time.contention.duration):
            contention_time = explainer.build_time_descriptor(
                "waiting time due to resource contention changed!",
                lambda event: event.waiting_time.contention.duration,
                parent=waiting_time,
            )

            # check changes in the arrival rate
            LOGGER.verbose("checking drifts in arrival rate")
            if explainer.has_drift_in_rate(
                    filter_=lambda event: event.activity == first_activity,
                    extractor=lambda event: event.enabled,
            ):
                explainer.build_rate_descriptor(
                    "arrival rates changed!",
                    parent=contention_time,
                    filter_=lambda event: event.activity == first_activity,
                    extractor=lambda event: event.enabled,
                )

            # check changes in the service rate
            LOGGER.verbose("checking drifts in service rate")
            if explainer.has_drift_in_rate(
                    filter_=lambda event: event.activity == last_activity,
                    extractor=lambda event: event.end,
            ):
                explainer.build_rate_descriptor(
                    "service rates changed!",
                    parent=contention_time,
                    filter_=lambda event: event.activity == last_activity,
                    extractor=lambda event: event.end,
                )

        # check waiting time due to prioritization
        LOGGER.verbose("checking drifts in prioritization waiting time")
        if explainer.has_drift_in_time(lambda event: event.waiting_time.prioritization.duration):
            prioritization_time = explainer.build_time_descriptor(
                "waiting time due to prioritization changed!",
                lambda event: event.waiting_time.prioritization.duration,
                parent=waiting_time,
            )

            # check changes in the prioritization policies
            LOGGER.verbose("checking drifts in prioritization policies")
            if explainer.has_drift_in_policies(build_prioritization_features):
                explainer.build_policies_descriptor(
                    "prioritization policies changed!",
                    parent=prioritization_time,
                    feature_extractor=build_prioritization_features,
                )

        # check waiting time due to unavailability
        LOGGER.verbose("checking drifts in unavailability waiting time")
        if explainer.has_drift_in_time(lambda event: event.waiting_time.availability.duration):
            unavailability_time = explainer.build_time_descriptor(
                "waiting time due to resource unavailability changed!",
                lambda event: event.waiting_time.availability.duration,
                parent=waiting_time,
            )

            # check changes in the availability calendars
            LOGGER.verbose("checking drifts in calendars")
            if explainer.has_drift_in_calendar():
                explainer.build_calendar_descriptor(
                    "resource availability calendars changed!",
                    parent=unavailability_time,
                )

        # check waiting time due to extraneous
        LOGGER.verbose("checking drifts in extraneous waiting time")
        if explainer.has_drift_in_time(lambda event: event.waiting_time.extraneous.duration):
            extraneous_time = explainer.build_time_descriptor(
                "waiting time due to extraneous delays changed!",
                lambda event: event.waiting_time.extraneous.duration,
                parent=waiting_time,
            )

            # check drifts in the activity profiles
            LOGGER.verbose("checking drifts in activity profiles")
            if explainer.has_drift_in_profile(ActivityProfile.discover):
                explainer.build_profile_descriptor(
                    "activity profiles changed!",
                    parent=extraneous_time,
                    profile_builder=ActivityProfile.discover,
                )
            # check drifts in the resource profiles
            LOGGER.verbose("checking drifts in resource profiles")
            if explainer.has_drift_in_profile(ResourceProfile.discover):
                explainer.build_profile_descriptor(
                    "resource profiles changed!",
                    parent=extraneous_time,
                    profile_builder=ResourceProfile.discover,
                )

    # create a tree with the change
    return root_cause
