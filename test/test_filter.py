from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.model import Log
from expert.utils.batching import discover_batches
from expert.utils.concurrency import HeuristicsConcurrencyOracle
from expert.utils.rules import Rule, Clause, filter_log


def __preprocess(event_log: Log) -> Log:

    event_log = list(event_log)

    for event in event_log:
        if event.resource == "Fake_Resource":
            event.resource = None

    return discover_batches(HeuristicsConcurrencyOracle(event_log).compute_enablement_timestamps())


files = (
    # ("batching1", "../data/logs/batches1.csv"),
    # ("batching2", "../data/logs/batches2.csv"),
    # ("batching3", "../data/logs/batches3.csv"),
    ("batching4", "../data/logs/batches4.csv"),
    # ("batching5", "../data/logs/batches5.csv"),
)

log = list(
    read_and_merge_csv_logs(
        files,
        attribute_mapping=EventMapping(
            start="start_time",
            end="end_time",
            case="case_id",
            activity="activity",
            resource="resource",
            attributes={"priority", "origin"}
        ),
        preprocessor=__preprocess,
        add_artificial_start_end_events=True
    ),
)

rule = Rule(
    clauses=frozenset([
        Clause(
            feature="activity",
            op="==",
            value="Sequence A"
        ),
        Clause(
            feature="attributes.priority",
            op="==",
            value="LOW"
        ),
    ]),
    reducer="&&"
)

filtered = filter_log(rule, log)

