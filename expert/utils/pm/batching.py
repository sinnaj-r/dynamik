import typing
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from expert.model import Batch, Log


@dataclass
class __BatchFiringState:
    # a class representing the firing state of a batch
    size: int
    time_since_first: float
    time_since_last: float
    hour_of_day: int
    minute_of_hour: int
    day_of_week: str
    fired: bool


@dataclass
class __BatchCreationState:
    # a class representing the batch creation state
    activity: str
    resource: str
    attributes: typing.Mapping[str, typing.Any]
    in_batch: bool


def discover_batches(log: Log) -> Log:
    """
    Compute the batches in the event log and add their descriptor to the events.

    Parameters
    ----------
    * `log`:   *an event log*

    Returns
    -------
    * the event log with the batches information
    """
    batches = []

    # Group events by resource and activity
    events_per_resource_and_activity = defaultdict(lambda: defaultdict(list))
    for event in log:
        if event.resource is not None:
            events_per_resource_and_activity[event.resource][event.activity].append(event)

    # Build batches for each resource and activity
    for resource in events_per_resource_and_activity:
        for activity in events_per_resource_and_activity[resource]:
            # Create a new empty batch (new activity implies new batch)
            current_batch = []

            for event in sorted(events_per_resource_and_activity[resource][activity], key=lambda evt: evt.enabled):
                # Add the event to the current batch if in first iteration or if it was enabled between the first event enablement and the
                # batch started executing
                if len(current_batch) == 0 or min(evt.enabled for evt in current_batch) <= event.enabled <= min(
                        evt.start for evt in current_batch):
                    current_batch.append(event)
                # If the event is not part of the current batch, save the current batch and create a new one with the
                # current event
                else:
                    batches.append(current_batch)
                    current_batch = [event]

            # Save the batch for the last iteration
            batches.append(current_batch)

    # Build batch descriptors and add them to the events
    for batch in batches:
        batch_descriptor = Batch(
            activity=batch[0].activity,
            resource=batch[0].resource,
            events=batch,
        )

        for event in batch:
            event.batch = batch_descriptor

    return log


def build_batch_firing_features(log: Log) -> pd.DataFrame:
    """Build the batch firing state from the log provided as argument"""
    # store the list of batch states
    states = []

    # get the events that belong to a batch
    batched_events = [event for event in log if event.batch is not None and event.batch.size > 1]

    for event in batched_events:
        # save the already-enabled events
        events_before_this = [evt for evt in event.batch.events if evt.enabled < event.enabled]

        # add the state of the batch for each event enablement timestamp
        states.append(
            __BatchFiringState(
                # the size is the number of events already enabled from the batch
                size=len([evt for evt in event.batch.events if evt.enabled <= event.enabled]),
                # the time elapsed since the first event in the batch has been enabled, in seconds
                time_since_first=0 if len(events_before_this) == 0 else (
                        event.enabled - min(evt.enabled for evt in events_before_this)).total_seconds(),
                # the time elapsed since the previous event in the batch has been enabled, in seconds
                time_since_last=0 if len(events_before_this) == 0 else (
                        event.enabled - max(evt.enabled for evt in events_before_this)).total_seconds(),
                # the hour of day for the enablement timestamp
                hour_of_day=event.enabled.hour,
                # the minute of the hour when the event was enabled
                minute_of_hour=event.enabled.minute,
                # the day of the week when the event was enabled
                day_of_week=event.enabled.strftime("%A"),
                # the state of the batch. consider a batch as fired when it starts executing
                fired=(event.enabled == event.batch.execution.begin) and (event.batch.size > 1),
            ),
        )

        # after evaluating the last event from the batch, add the batch state when the batch is fired
        if event.enabled == event.batch.accumulation.end:
            batch = event.batch

            states.append(
                __BatchFiringState(
                    # since is the last event from the batch, the batch size is the total count of events in the batch
                    size=len(list(batch.events)),
                    # the time since the first event was enabled and the batch started executing
                    time_since_first=(batch.execution.begin - min(evt.enabled for evt in batch.events)).total_seconds(),
                    # the time since the last event was enabled and the batch started executing
                    time_since_last=(batch.execution.begin - max(evt.enabled for evt in batch.events)).total_seconds(),
                    # the hour of the day the batch started executing
                    hour_of_day=batch.execution.begin.hour,
                    # the minute of the hour the batch started executing
                    minute_of_hour=batch.execution.begin.minute,
                    # the day of week the batch started executing
                    day_of_week=batch.execution.begin.strftime("%A"),
                    # since the timestamp is the start of the execution, the state of the batch should be fired
                    fired=batch.size > 1,
                ),
            )

    # transform the list of states to a pandas dataframe
    features = pd.DataFrame.from_records([state.__dict__ for state in states]).infer_objects()
    # set correct type to categorical columns
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns
    # return the features dataframe with correct types
    return (pd.concat([features.drop(categorical_columns, axis=1), features[categorical_columns].astype("category")], axis=1)
            .rename(columns={"fired": "class"})).drop_duplicates()


def build_batch_creation_features(log: Log) -> pd.DataFrame:
    """Build the batch creation features from the log provided as argument"""
    # store the list of batch states
    states = []

    for event in log:
        # filter out synthetic events
        if event.activity not in ("__SYNTHETIC_START_EVENT__", "__SYNTHETIC_END_EVENT__"):
            # add the state of the batch for each event
            states.append(
                __BatchCreationState(
                    activity=event.activity,
                    resource=event.resource,
                    attributes=event.attributes,
                    in_batch=event.batch is not None and event.batch.size > 1,
                ),
            )

    # transform the list of states to a pandas dataframe
    features = pd.json_normalize([state.__dict__ for state in states]).infer_objects()
    # set correct type to categorical columns
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns
    # return the features dataframe with correct types
    return (pd.concat([features.drop(categorical_columns, axis=1), features[categorical_columns].astype("category")], axis=1)
            .rename(columns={"in_batch": "class"})).drop_duplicates()
