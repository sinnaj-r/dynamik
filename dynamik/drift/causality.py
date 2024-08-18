from __future__ import annotations

import itertools
import typing
from datetime import datetime, timedelta
from statistics import mean

import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import ttost_ind

from dynamik.drift.model import Drift, DriftCause
from dynamik.model import Event, Log, Resource
from dynamik.utils.logger import LOGGER
from dynamik.utils.model import DistributionDescription, HashableDF, Pair
from dynamik.utils.pm.batching import build_batch_creation_features, build_batch_firing_features
from dynamik.utils.pm.calendars import Calendar, discover_calendars
from dynamik.utils.pm.prioritization import build_prioritization_features
from dynamik.utils.pm.profiles import ActivityProfile, Profile, ResourceProfile
from dynamik.utils.rules import ConfusionMatrix, Rule, compute_rule_score, discover_rules, filter_log


class DriftExplainer:
    """TODO docs"""

    drift: Drift
    significance: float
    calendar_threshold: float
    threshold: timedelta | float

    def __init__(
            self: typing.Self,
            drift: Drift,
            significance: float,
            threshold: timedelta | float,
            calendar_threshold: float,
    ) -> None:
        self.drift = drift
        self.significance = significance
        self.calendar_threshold = calendar_threshold
        self.threshold = threshold

    def __describe_distributions(
            self: typing.Self,
            extractor: typing.Callable[[Event], timedelta],
    ) -> Pair[DistributionDescription]:
        return Pair(
            reference=DistributionDescription(
                scipy.stats.describe(
                    [extractor(event).total_seconds() for event in self.drift.reference_model.data if event.resource is not None],
                ),
            ),
            running=DistributionDescription(
                scipy.stats.describe(
                    [extractor(event).total_seconds() for event in self.drift.running_model.data if event.resource is not None],
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
                reference_calendars.values(),
                Calendar(),
            ),
            running=sum(
                running_calendars.values(),
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
            results.reference[repr(rule)] = compute_rule_score(rule, HashableDF(reference_features))
            results.running[repr(rule)] = compute_rule_score(rule, HashableDF(running_features))

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
            reference={
                owner: calendar.asdict() for (owner, calendar) in discover_calendars(self.drift.reference_model.data).items()
            },
            running={
                owner: calendar.asdict() for (owner, calendar) in discover_calendars(self.drift.running_model.data).items()
            },
        )

    def __get_time_data(
            self: typing.Self,
            time_extractor: typing.Callable[[Event], timedelta],
    ) -> Pair[typing.Iterable[timedelta]]:
        """TODO docs"""
        return Pair(
            reference=[time_extractor(event) for event in self.drift.reference_model.data if event.resource is not None],
            running=[time_extractor(event) for event in self.drift.running_model.data if event.resource is not None],
        )

    def __get_rate_data(
            self: typing.Self,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> Pair[Calendar]:
        # get the timestamps
        ref_times: list[datetime] = sorted([extractor(event) for event in self.drift.reference_model.data if filter_(event)])
        run_times: list[datetime] = sorted([extractor(event) for event in self.drift.running_model.data if filter_(event)])
        # compute inter-rate times
        ref_inter_times = [(t2 - t1).total_seconds() for (t1, t2) in itertools.pairwise(ref_times)]
        run_inter_times = [(t2 - t1).total_seconds() for (t1, t2) in itertools.pairwise(run_times)]
        # compute mean inter-rate times
        avg_ref_inter_time = mean(ref_inter_times)
        avg_run_inter_time = mean(run_inter_times)
        # rates are 1/inter-rate-time. inter-rate-time is in seconds, so we multiply by 3600 for translating to instances/hour
        return Pair(
            reference=3600/avg_ref_inter_time,
            running=3600/avg_run_inter_time,
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
        if not self.drift.reference_model.empty and not self.drift.running_model.empty:
            reference_data = [time_extractor(event).total_seconds() for event in self.drift.reference_model.data]
            running_data = [time_extractor(event).total_seconds() for event in self.drift.running_model.data]
            t = self.threshold

            if isinstance(self.threshold, float):
                scaler = StandardScaler()
                scaler.fit(np.array(reference_data).reshape(-1, 1))
                reference_data = scaler.transform(np.array(reference_data).reshape(-1, 1)).flatten()
                running_data = scaler.transform(np.array(running_data).reshape(-1, 1)).flatten()
            else:
                t = self.threshold.total_seconds()

            # only if both models are non-empty, perform the test
            pvalue, _, _ = ttost_ind(reference_data, running_data, -t, t)

            LOGGER.verbose('test(reference != running) p-value: %.4f', pvalue)

            return pvalue > self.significance

        return False

    def has_drift_in_calendar(
            self: typing.Self,
    ) -> bool:
        """TODO docs"""
        # discover calendars for reference and running models
        reference_calendars = discover_calendars(self.drift.reference_model.data)
        running_calendars = discover_calendars(self.drift.running_model.data)

        resources = set(reference_calendars.keys()).union(running_calendars.keys())

        # check if calendars are equivalent for each resource
        for resource in resources:
            if resource in reference_calendars and resource in running_calendars:
                if not reference_calendars[resource].equivalent(running_calendars[resource], self.calendar_threshold):
                    return True
            else:
                return True

        return False

    def has_drift_in_rate(
            self: typing.Self,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> bool:
        """TODO docs"""
        # compute the average reference rate distribution per hour and day of week
        reference_rates = Calendar.discover(
            [event for event in self.drift.reference_model.data if filter_(event)],
            lambda event: [extractor(event)],
        )

        # compute the average running arrival distribution per hour and day of week
        running_rates = Calendar.discover(
            [event for event in self.drift.running_model.data if filter_(event)],
            lambda event: [extractor(event)],
        )

        reference_size = sum(reference_rates[slot] for slot in reference_rates.slots)
        running_size = sum(running_rates[slot] for slot in running_rates.slots)

        # normalize with the absolute count to get the frequency
        reference_rates = reference_rates.transform(lambda value: value/reference_size)
        running_rates = running_rates.transform(lambda value: value/running_size)

        if reference_size == 0 or running_size == 0:
            LOGGER.warning('can not check rate in models')
            LOGGER.warning('no cases start or end in this window')
            LOGGER.warning('try increasing the window size')
            return False

        pvalue, _, _ = ttost_ind(reference_rates.values, running_rates.values, -0.1, 0.1)

        return pvalue > self.significance

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

            pvalue, _, _ = ttost_ind([score.f1_score for score in reference_scores], [score.f1_score for score in running_scores], 0, 0)
            results.append(pvalue)
            LOGGER.verbose("test(reference != running) p-value: %.4f", pvalue)

        return any(pvalue > self.significance for pvalue in results)

    def has_drift_in_profile(
            self: typing.Self,
            profile_builder: typing.Callable[[Log], Profile],
    ) -> bool:
        """TODO docs"""
        reference_profile = profile_builder(self.drift.reference_model.data)
        running_profile = profile_builder(self.drift.running_model.data)

        return not reference_profile.statistically_equals(running_profile, self.significance)

    def build_time_descriptor(
            self: typing.Self,
            what: str,
            time_extractor: typing.Callable[[Event], timedelta],
            parent: DriftCause | None = None,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed?
            what=what,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_distributions(time_extractor),
            # data contains the raw data used in the test
            data=self.__get_time_data(time_extractor),
            # the causes of the drift are the changes in the effective and the idle processing times
            parent=parent,
        )

    def build_calendar_descriptor(
            self: typing.Self,
            what: str,
            parent: DriftCause | None = None,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? the processing time
            what=what,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_calendars(),
            # data contains the raw data used in the test
            data=self.__get_calendars(),
            # the causes of the drift are the changes in the effective and the idle processing times
            parent=parent,
        )

    def build_rate_descriptor(
            self: typing.Self,
            what: str,
            parent: DriftCause | None = None,
            *,
            filter_: typing.Callable[[Event], bool],
            extractor: typing.Callable[[Event], datetime],
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? the processing time
            what=what,
            # how did it change? include the distributions for both pre- and post- drift data
            how=self.__describe_rates(filter_, extractor),
            # data contains the raw data used in the test
            data=self.__get_rate_data(filter_, extractor),
            # the causes of the drift are the changes in the effective and the idle processing times
            parent=parent,
        )

    def build_policies_descriptor(
            self: typing.Self,
            what: str,
            parent: DriftCause | None = None,
            *,
            feature_extractor: typing.Callable[[Log], pd.DataFrame],
            filter_: typing.Callable[[Log], Log] = lambda _: _,
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? policies
            what=what,
            # how did it change? include the policies for both pre- and post- drift data
            how=self.__describe_policies(feature_extractor, filter_),
            # data contains the evaluation of policies for before and after
            data=self.__describe_policies(feature_extractor, filter_),
            parent=parent,
        )

    def build_profile_descriptor(
            self: typing.Self,
            what: str,
            parent: DriftCause | None = None,
            *,
            profile_builder: typing.Callable[[Log], Profile],
    ) -> DriftCause:
        """TODO docs"""
        return DriftCause(
            # what changed? policies
            what=what,
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
        threshold: timedelta | float = timedelta(minutes=1),
        calendar_threshold: float = 0.0,
) -> DriftCause:
    """Build a tree with the causes that explain the drift characterized by the given drift features"""
    # if there is a drift in the cycle time distribution, check for drifts in the waiting and processing times and build
    # a tree accordingly, explaining the changes that occurred to the process
    explainer = DriftExplainer(drift, significance, threshold, calendar_threshold)
    root_cause = explainer.build_time_descriptor(
        what='cycle-time',
        time_extractor=lambda event: event.cycle_time,
    )

    # check processing time
    LOGGER.verbose('checking drifts in total processing time')
    if explainer.has_drift_in_time(lambda event: event.processing_time.total.duration):
        # add a node to the tree reporting the change in the processing times
        processing_time = explainer.build_time_descriptor(
            what=f'{root_cause.what}/processing-time',
            time_extractor=lambda event: event.processing_time.total.duration,
            parent=root_cause,
        )

        # check drift in processing time when the resource is available
        LOGGER.verbose('checking drifts in effective processing time')
        if explainer.has_drift_in_time(lambda event: event.processing_time.effective.duration):
            effective_time = explainer.build_time_descriptor(
                what=f'{processing_time.what}/available',
                time_extractor=lambda event: event.processing_time.effective.duration,
                parent=processing_time,
            )

            # check drifts in the activity profiles
            LOGGER.verbose('checking drifts in activity profiles')
            if explainer.has_drift_in_profile(ActivityProfile.discover):
                explainer.build_profile_descriptor(
                    what=f'{effective_time.what}/activity-profiles',
                    parent=effective_time,
                    profile_builder=ActivityProfile.discover,
                )
            # check drifts in the resource profiles
            LOGGER.verbose('checking drifts in resource profiles')
            if explainer.has_drift_in_profile(ResourceProfile.discover):
                explainer.build_profile_descriptor(
                    what=f'{effective_time.what}/resource-profiles',
                    parent=effective_time,
                    profile_builder=ResourceProfile.discover,
                )

        # check drift in processing time when the resource is not available
        LOGGER.verbose('checking drifts in idle processing time')
        if explainer.has_drift_in_time(lambda event: event.processing_time.idle.duration):
            idle_time = explainer.build_time_descriptor(
                what=f'{processing_time.what}/unavailable',
                time_extractor=lambda event: event.processing_time.idle.duration,
                parent=processing_time,
            )

            # check changes in the availability calendars
            LOGGER.verbose('checking drifts in calendars')
            if explainer.has_drift_in_calendar():
                explainer.build_calendar_descriptor(
                    what=f'{idle_time.what}/calendars',
                    parent=idle_time,
                )

    # check waiting time
    LOGGER.verbose('checking drifts in total waiting time')
    if explainer.has_drift_in_time(lambda event: event.waiting_time.total.duration):
        # add a node to the tree reporting the change in the waiting times
        waiting_time = explainer.build_time_descriptor(
            what=f'{root_cause.what}/waiting-time',
            time_extractor=lambda event: event.waiting_time.total.duration,
            parent=root_cause,
        )

        # check waiting time due to batching
        LOGGER.verbose('checking drifts in batching waiting time')
        if explainer.has_drift_in_time(lambda event: event.waiting_time.batching.duration):
            batching_time = explainer.build_time_descriptor(
                what=f'{waiting_time.what}/batching',
                time_extractor=lambda event: event.waiting_time.batching.duration,
                parent=waiting_time,
            )

            # check changes in the batch creation policies
            LOGGER.verbose('checking drifts batch creation policies')
            if explainer.has_drift_in_policies(build_batch_creation_features):
                explainer.build_policies_descriptor(
                    what=f'{batching_time.what}/batch-creation',
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
                LOGGER.verbose('checking drifts in batch firing policies')
                if explainer.has_drift_in_policies(build_batch_firing_features, filter_log(creation_policy)):
                    explainer.build_policies_descriptor(
                        what=f'{batching_time.what}/batch-firing',
                        parent=batching_time,
                        feature_extractor=build_batch_firing_features,
                        filter_=filter_log(creation_policy),
                    )

        # check waiting time due to contention
        LOGGER.verbose('checking drifts in contention waiting time')
        if explainer.has_drift_in_time(lambda event: event.waiting_time.contention.duration):
            contention_time = explainer.build_time_descriptor(
                what=f'{waiting_time.what}/contention',
                time_extractor=lambda event: event.waiting_time.contention.duration,
                parent=waiting_time,
            )

            # check changes in the arrival rate
            LOGGER.verbose('checking drifts in arrival rate')
            if explainer.has_drift_in_rate(
                    filter_=lambda event: event.activity == first_activity,
                    extractor=lambda event: event.enabled,
            ):
                explainer.build_rate_descriptor(
                    what=f'{contention_time.what}/arrival-rates',
                    parent=contention_time,
                    filter_=lambda event: event.activity == first_activity,
                    extractor=lambda event: event.enabled,
                )

            # check changes in the service rate
            LOGGER.verbose('checking drifts in service rate')
            if explainer.has_drift_in_rate(
                    filter_=lambda event: event.activity == last_activity,
                    extractor=lambda event: event.end,
            ):
                explainer.build_rate_descriptor(
                    what=f'{contention_time.what}/service-rates',
                    parent=contention_time,
                    filter_=lambda event: event.activity == last_activity,
                    extractor=lambda event: event.end,
                )

        # check waiting time due to prioritization
        LOGGER.verbose('checking drifts in prioritization waiting time')
        if explainer.has_drift_in_time(lambda event: event.waiting_time.prioritization.duration):
            prioritization_time = explainer.build_time_descriptor(
                what=f'{waiting_time.what}/prioritization',
                time_extractor=lambda event: event.waiting_time.prioritization.duration,
                parent=waiting_time,
            )

            # check changes in the prioritization policies
            LOGGER.verbose('checking drifts in prioritization policies')
            if explainer.has_drift_in_policies(build_prioritization_features):
                explainer.build_policies_descriptor(
                    what=f'{prioritization_time.what}/prioritization-policies',
                    parent=prioritization_time,
                    feature_extractor=build_prioritization_features,
                )

        # check waiting time due to unavailability
        LOGGER.verbose('checking drifts in unavailability waiting time')
        if explainer.has_drift_in_time(lambda event: event.waiting_time.availability.duration):
            unavailability_time = explainer.build_time_descriptor(
                what=f'{waiting_time.what}/unavailability',
                time_extractor=lambda event: event.waiting_time.availability.duration,
                parent=waiting_time,
            )

            # check changes in the availability calendars
            LOGGER.verbose('checking drifts in calendars')
            if explainer.has_drift_in_calendar():
                explainer.build_calendar_descriptor(
                    what=f'{unavailability_time.what}/calendars',
                    parent=unavailability_time,
                )

        # check waiting time due to extraneous
        LOGGER.verbose('checking drifts in extraneous waiting time')
        if explainer.has_drift_in_time(lambda event: event.waiting_time.extraneous.duration):
            extraneous_time = explainer.build_time_descriptor(
                what=f'{waiting_time.what}/extraneous',
                time_extractor=lambda event: event.waiting_time.extraneous.duration,
                parent=waiting_time,
            )

            # check drifts in the activity profiles
            LOGGER.verbose('checking drifts in activity profiles')
            if explainer.has_drift_in_profile(ActivityProfile.discover):
                explainer.build_profile_descriptor(
                    what=f'{extraneous_time.what}/activity-profiles',
                    parent=extraneous_time,
                    profile_builder=ActivityProfile.discover,
                )
            # check drifts in the resource profiles
            LOGGER.verbose('checking drifts in resource profiles')
            if explainer.has_drift_in_profile(ResourceProfile.discover):
                explainer.build_profile_descriptor(
                    what=f'{extraneous_time.what}/resource-profiles',
                    parent=extraneous_time,
                    profile_builder=ResourceProfile.discover,
                )

    # create a tree with the change
    return root_cause
