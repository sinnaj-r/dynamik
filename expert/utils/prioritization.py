import typing

import pandas as pd

from expert.model import Log
from expert.utils.rules import Rule, discover_rules


def __find_prioritized_events(log: Log) -> list[dict]:
    # a list of pairs (delayed event, prioritized event)
    prioritized = []

    # create a list of dicts representing the event log
    events_as_dict = [event.asdict() for event in log]

    # build two dataframes with "reference" and "prioritized" events
    reference_events = pd.json_normalize(events_as_dict).add_prefix("reference.")
    alternative_events = pd.json_normalize(events_as_dict).add_prefix("prioritized.")

    event_pairs = reference_events.join(alternative_events, how="cross")

    # build a rule for filtering the pairs that show prioritization (those that have been enabled after the reference
    # one and started executing before it did)
    query = ((event_pairs["reference.resource"] == event_pairs["prioritized.resource"]) &
             (event_pairs["reference.enabled"] < event_pairs["prioritized.enabled"]) &
             (event_pairs["reference.start"] > event_pairs["prioritized.start"]))

    # get the list of prioritized event pairs
    prioritized_events = event_pairs.loc[query, :]

    # build the features list from the dataframe
    for _, row in prioritized_events.iterrows():
        prioritized.append({
            "resource": row["prioritized.resource"],
            "activity": row["prioritized.activity"],
            "attributes": {
                attr.replace("prioritized.attributes.", ""): row[attr] for attr in row.axes[0].tolist() if attr.startswith("prioritized.attributes")
            },
            "prioritized": True,
        })

    return prioritized


def __find_non_prioritized_events(log: Log) -> list[dict]:
    # a list of pairs (delayed event, prioritized event)
    non_prioritized = []

    # create a list of dicts representing the event log
    events_as_dict = [event.asdict() for event in log]

    # build two dataframes with "reference" and "prioritized" events
    reference_events = pd.json_normalize(events_as_dict).add_prefix("reference.")
    alternative_events = pd.json_normalize(events_as_dict).add_prefix("prioritized.")

    event_pairs = reference_events.join(alternative_events, how="cross")

    # build a rule for filtering the pairs that do not show prioritization (those that have been enabled before the reference
    # one and also started executing before it did)
    query = ((event_pairs["reference.resource"] == event_pairs["prioritized.resource"]) &
             (event_pairs["reference.enabled"] < event_pairs["prioritized.enabled"]) &
             (event_pairs["reference.start"] > event_pairs["prioritized.enabled"]) &
             (event_pairs["reference.start"] < event_pairs["prioritized.start"]))

    # get the list of non-prioritized event pairs
    non_prioritized_events = event_pairs.loc[query, :]

    # build the features list from the dataframe
    for _, row in non_prioritized_events.iterrows():
        non_prioritized.append({
            "resource": row["prioritized.resource"],
            "activity": row["prioritized.activity"],
            "attributes": {
                attr.replace("prioritized.attributes.", ""): row[attr] for attr in row.axes[0].tolist() if attr.startswith("prioritized.attributes")
            },
            "prioritized": False,
        })

    return non_prioritized


def build_prioritization_features(log: Log) -> pd.DataFrame:
    """Build the matrix of features for prioritization from the given event log"""
    # find the prioritized and non prioritized pairs of events
    prioritized_events = __find_prioritized_events(log)
    non_prioritized_events = __find_non_prioritized_events(log)

    # build the features for the prioritized events
    prioritized_features = pd.json_normalize(prioritized_events)
    # build the features for the non-prioritized events
    non_prioritized_features = pd.json_normalize(non_prioritized_events)
    # concat both prioritized and non prioritized events in a single dataframe
    features = pd.concat([prioritized_features, non_prioritized_features], ignore_index=True).infer_objects()
    # set correct type to categorical columns
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns
    # return the features dataframe with correct types
    return pd.concat([features.drop(categorical_columns, axis=1), features[categorical_columns].astype("category")], axis=1)


def discover_prioritization_policies(log: Log, min_precision: float = 0.9, min_recall: float = 0.01) -> typing.Iterable[Rule]:
    """TODO"""
    # build the features for prioritizing and delaying events
    prioritization_features = build_prioritization_features(log)

    # return the rules modeling the priorities
    return discover_rules(
        prioritization_features,
        class_attr="prioritized",
        encode_categorical=True,
        balance_data=True,
        min_rule_recall=min_recall,
        min_rule_precision=min_precision,
    )
