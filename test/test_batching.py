
from expert.input import EventMapping
from expert.input.csv import read_and_merge_csv_logs
from expert.model import Log
from expert.utils.batching import compute_batches, discover_batch_creation_policies, discover_batch_firing_policies
from expert.utils.rules import filter_log


def __preprocess(event_log: Log) -> Log:

    event_log = list(event_log)

    for event in event_log:
        if event.resource == "Fake_Resource":
            event.resource = None

    return compute_batches(event_log)


files = (
    ("batching0", "../data/logs/base-sequence-batching.csv"),
    ("batching1", "../data/logs/batches1.csv"),
    ("batching2", "../data/logs/batches2.csv"),
    ("batching3", "../data/logs/batches3.csv"),
    ("batching4", "../data/logs/batches4.csv"),
    ("batching5", "../data/logs/batches5.csv"),
    # ("base", "../data/logs/base-sequence.csv"),
    # ("batching", "../data/logs/base-sequence-batching.csv"),
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
            enablement="enable_time",
            attributes={"priority", "origin"},
        ),
        preprocessor=__preprocess,
        add_artificial_start_end_events=False,
    ),
)

creation_rules = discover_batch_creation_policies(log, min_precision=0.9)

for (i, rule) in enumerate(creation_rules):
    print(f"Creation rule {i}: {rule.__repr__()}")
    print(f"    Precision: {rule.score.precision}")
    print(f"    Recall:    {rule.score.recall}")
    print(f"    Accuracy:  {rule.score.classification_accuracy}")
    print(f"    F1:        {rule.score.f1_score}")
    print()

    firing_rules = discover_batch_firing_policies(filter_log(rule, log))
    for (j, firing_rule) in enumerate(firing_rules):
        print(f"Firing rule {i}.{j}: {firing_rule.__repr__()}")
        print(f"    Precision: {firing_rule.score.precision}")
        print(f"    Recall:    {firing_rule.score.recall}")
        print(f"    Accuracy:  {firing_rule.score.classification_accuracy}")
        print(f"    F1:        {firing_rule.score.f1_score}")

    print()
    print("--------------------------------------------------------------------------")
    print()
