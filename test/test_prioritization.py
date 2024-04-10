from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.model import Log
from expert.utils.batching import compute_batches
from expert.utils.concurrency import HeuristicsConcurrencyOracle
from expert.utils.prioritization import discover_prioritization_policies


def __preprocess(event_log: Log) -> Log:

    event_log = list(event_log)

    for event in event_log:
        if event.resource == "Fake_Resource":
            event.resource = None

    return compute_batches(HeuristicsConcurrencyOracle(event_log).compute_enablement_timestamps())


files = (
    # ("priority1", "../data/logs/priority1.csv"),
    # ("priority2", "../data/logs/priority2.csv"),
    ("priority3", "../data/logs/priority3.csv"),
)

log = list(
    read_and_merge_csv_logs(
        files,
        attribute_mapping=EventMapping(
            start="start_time",
            end="end_time",
            enablement="enabled_time",
            case="case_id",
            activity="activity",
            resource="resource",
            # attributes={"loan_amount"},
            attributes={"urgency"},
        ),
    ),
)

priorities = discover_prioritization_policies(log)

for priority in priorities:
    print(f"{priority.__repr__()}: {priority.score.f1_score}")

print()
