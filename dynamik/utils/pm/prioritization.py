import functools

import janitor
import pandas as pd

from dynamik.model import Event, Log


def __event_as_dict(event: Event) -> dict:
    return {
        "case": event.case,
        "activity": event.activity,
        "resource": event.resource,
        "start": event.start,
        "enabled": event.enabled,
        "attributes": event.attributes,
    }


def __find_prioritized_events(log: Log) -> list[dict]:
    # a list of pairs (delayed event, prioritized event)
    prioritized = []

    # create a list of dicts representing the event log
    events_as_dict = [__event_as_dict(event) for event in log if event.resource is not None]

    # build two dataframes with "reference" and "prioritized" events
    events = pd.json_normalize(events_as_dict)
    events["start"] = events["start"].dt.tz_convert(None)
    events["enabled"] = events["enabled"].dt.tz_convert(None)

    reference_events = events.add_prefix("reference.")
    alternative_events = events.add_prefix("prioritized.")

    # merge events with that show prioritization (those that have been executed by the same resource,
    # enabled after the reference one and started executing before it did)
    prioritized_events = reference_events.conditional_join(
        alternative_events,
        janitor.col("reference.resource") == janitor.col("prioritized.resource"),
        janitor.col("reference.enabled") < janitor.col("prioritized.enabled"),
        janitor.col("reference.start") > janitor.col("prioritized.start"),
    )

    # build the features list from the dataframe
    for _, row in prioritized_events.iterrows():
        prioritized.append({
            "resource": row["prioritized.resource"],
            "activity": row["prioritized.activity"],
            "attributes": {
                attr.replace("prioritized.attributes.", ""): row[attr] for attr in row.axes[0].tolist() if attr.startswith("prioritized.attributes")
            },
            "class": True,
        })

    return prioritized


def __find_non_prioritized_events(log: Log) -> list[dict]:
    # a list of pairs (delayed event, prioritized event)
    non_prioritized = []

    # create a list of dicts representing the event log
    events_as_dict = [__event_as_dict(event) for event in log if event.resource is not None]

    # build two dataframes with "reference" and "prioritized" events
    events = pd.json_normalize(events_as_dict)
    events["start"] = events["start"].dt.tz_convert(None)
    events["enabled"] = events["enabled"].dt.tz_convert(None)

    reference_events = events.add_prefix("reference.")
    alternative_events = events.add_prefix("prioritized.")

    # merge events with that do not show prioritization (those that have been executed by the same resource,
    # enabled after the reference one and started executing after it did)
    non_prioritized_events = reference_events.conditional_join(
        alternative_events,
        janitor.col("reference.resource") == janitor.col("prioritized.resource"),
        janitor.col("reference.enabled") < janitor.col("prioritized.enabled"),
        janitor.col("reference.start") > janitor.col("prioritized.enabled"),
        janitor.col("reference.start") < janitor.col("prioritized.start"),
    )

    # build the features list from the dataframe
    for _, row in non_prioritized_events.iterrows():
        non_prioritized.append({
            "resource": row["prioritized.resource"],
            "activity": row["prioritized.activity"],
            "attributes": {
                attr.replace("prioritized.attributes.", ""): row[attr] for attr in row.axes[0].tolist() if attr.startswith("prioritized.attributes")
            },
            "class": False,
        })

    return non_prioritized


@functools.lru_cache
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
    return pd.concat([features.drop(categorical_columns, axis=1), features[categorical_columns].astype("category")], axis=1).drop_duplicates()
