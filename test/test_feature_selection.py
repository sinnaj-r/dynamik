import random
from datetime import UTC, datetime, timedelta

import scipy
from pandas import CategoricalDtype

from expert.model import Event, TimeInterval, Log, WaitingTime
from expert.utils.feature_selection import chained_selectors, from_model, select_relevant_features, univariate
from expert.utils.statistical_tests import categorical_test, test

fake_log_1 = [
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "LOW",
            "complexity": "LOW",
            "cost": random.randint(100, 5000),
            "benefit": random.randint(100, 500) * random.randint(1, 10),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(5),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "MED",
            "complexity": "LOW",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(1000, 2000) * random.randint(1, 10),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(3),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "HIGH",
            "complexity": "LOW",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(3000, 5000) * random.randint(1, 10),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(2),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "LOW",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(100, 500) * random.randint(5, 20),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(17),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "MED",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(1000, 2000) * random.randint(5, 20),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(15),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "HIGH",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(3000, 5000) * random.randint(5, 20),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(12),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "LOW",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(100, 500) * random.randint(10, 50),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(57),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "MED",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(1000, 2000) * random.randint(10, 50),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(55),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "HIGH",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(3000, 5000) * random.randint(10, 50),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(51),
            ),
        ),
    ),
]
fake_log_2 = [
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "LOW",
            "complexity": "LOW",
            "cost": random.randint(100, 5000),
            "benefit": random.randint(100, 500) * random.randint(1, 10),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(5),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "MED",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(1000, 2000) * random.randint(1, 10),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(3),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "HIGH",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(3000, 5000) * random.randint(1, 10),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(2),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "LOW",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(100, 500) * random.randint(5, 20),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(17),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "MED",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(1000, 2000) * random.randint(5, 20),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(15),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "HIGH",
            "complexity": "MED",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(3000, 5000) * random.randint(5, 20),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(12),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "LOW",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(100, 500) * random.randint(10, 50),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(57),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "MED",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(1000, 2000) * random.randint(10, 50),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(55),
            ),
        ),
    ),
    Event(
        case="",
        activity="",
        resource="",
        start=datetime.now(tz=UTC),
        end=datetime.now(tz=UTC),
        attributes={
            "priority": "HIGH",
            "complexity": "HIGH",
            "cost": random.randint(1000, 5000),
            "benefit": random.randint(3000, 5000) * random.randint(10, 50),
            "origin": "SPAIN",
            "order": 1.456,
        },
        waiting_time=WaitingTime(
            extraneous=TimeInterval(
                duration=timedelta(51),
            ),
        ),
    ),
]


complexity_mapping = {
    "LOW": 1,
    "MED": 2,
    "HIGH": 3,
}
priority_mapping = {
    "LOW": 1,
    "MED": 2,
    "HIGH": 3,
}


def __preprocess(log: Log) -> Log:
    for event in log:
        # if event.attributes["complexity"] in complexity_mapping:
        #     event.attributes["complexity"] = complexity_mapping[event.attributes["complexity"]]
        # if event.attributes["priority"] in priority_mapping:
        #     event.attributes["priority"] = priority_mapping[event.attributes["priority"]]
        event.attributes = {}

    return log


selected_features_1, features_1 = select_relevant_features(
    __preprocess([*fake_log_1, *fake_log_1, *fake_log_1, *fake_log_1, *fake_log_2, *fake_log_1, *fake_log_2]),
    class_extractor=lambda event: event.waiting_time.extraneous.duration,
    predictors_extractor=lambda event: event.attributes,
    feature_selector=chained_selectors([univariate(), from_model()]),
)
selected_features_2, features_2 = select_relevant_features(
    __preprocess([*fake_log_2, *fake_log_2, *fake_log_2, *fake_log_2, *fake_log_2, *fake_log_2, *fake_log_2]),
    class_extractor=lambda event: event.waiting_time.extraneous.duration,
    predictors_extractor=lambda event: event.attributes,
    feature_selector=chained_selectors([univariate(), from_model()]),
)

print("Analysis using Univariate+LASSO")
print(f"Relevant features are { {*selected_features_1, *selected_features_2} }")

for feature in {*selected_features_1, *selected_features_2}:
    if isinstance(features_1[feature].dtype, CategoricalDtype):
        if categorical_test(features_1[feature].tolist(), features_2[feature].tolist()):
            print(f"    Found significant differences in the distribution of feature {feature}")
            print(f"        Reference data was {features_1[feature].value_counts(normalize=True, sort=False).to_dict()}")
            print(f"        Observed data was {features_2[feature].value_counts(normalize=True, sort=False).to_dict()}")

    elif test(features_1[feature].tolist(), features_2[feature].tolist()):
        print(f"    Found significant differences in the distribution of feature {feature}")
        print(f"        Reference data was {scipy.stats.describe(features_1[feature])}")
        print(f"        Observed data was {scipy.stats.describe(features_2[feature])}")
