import functools
import operator
import re
import typing
from dataclasses import dataclass, field
from operator import itemgetter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imodels import SkopeRulesClassifier
from sklearn.preprocessing import OneHotEncoder

from expert.model import Event, Log

_operators = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "&&": lambda df: df.all(axis="columns"),
    "and": lambda df: df.all(axis="columns"),
    "||": lambda df: df.any(axis="columns"),
    "or": lambda df: df.any(axis="columns"),
}


@dataclass(frozen=True)
class ConfusionMatrix:
    """A confusion matrix"""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def observations(self: typing.Self) -> int:
        """The total number of observations in this confusion matrix"""
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives

    @property
    def precision(self: typing.Self) -> float:
        """The precision computes how many detected positives are real"""
        return self.true_positives / div if (div := self.true_positives+self.false_positives) != 0 else 0

    @property
    def recall(self: typing.Self) -> float:
        """The recall computes how many real positives are detected"""
        return self.true_positives / div if (div := self.true_positives+self.false_negatives) != 0 else 0

    @property
    def classification_accuracy(self: typing.Self) -> float:
        """The classification accuracy measures the number of correct predictions vs the total number of predictions made"""
        return (self.true_positives + self.true_negatives) / self.observations if self.observations != 0 else 0

    @property
    def f1_score(self: typing.Self) -> float:
        """F1 score measures the harmonic mean between precision and recall"""
        return 2 / ((1 / self.precision) + (1 / self.recall)) if self.precision != 0 and self.recall != 0 else 0

    def asdict(self: typing.Self) -> dict:
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "f1_score": self.f1_score,
            "classification_accuracy": self.classification_accuracy,
        }


@dataclass(frozen=True)
class Clause:
    """TODO"""

    feature: str
    op: str
    value: typing.Any

    def evaluate(
            self: typing.Self,
            data: pd.DataFrame,
            *,
            add_eval_to_original_dataframe: bool = False,
    ) -> pd.Series:
        """Evaluates the given rule over the given dataframe, returning a pd.Series with the evaluation result for each row"""
        # if the feature of the clause is present in the data, evaluate the clause
        if self.feature in data.columns:
            # eval the clause against the provided dataframe row by row
            evaluation = _operators[self.op](data.loc[:, self.feature], self.value)
        # otherwise, just create a new pd.Series with False values (as the rule can not be evaluated)
        else:
            evaluation = pd.Series(data=False, index=data.index)

        # set the clause string representation as the pd.Series name
        evaluation = evaluation.rename(self.__repr__())

        # if asked, add the result of the evaluation to the input dataframe as a new column
        if add_eval_to_original_dataframe:
            data[self.__repr__()] = evaluation
        # return the result of the evaluation
        return evaluation

    def replace(self: typing.Self, pattern: str, replacement: str) -> typing.Self:
        """Replace the given pattern with the replacement in the clause feature"""
        return Clause(
            feature=self.feature.replace(pattern, replacement),
            op=self.op,
            value=self.value,
        )

    def asdict(self: typing.Self) -> dict:
        return {
            "feature": self.feature,
            "op": self.op,
            "value": self.value,
        }

    def __repr__(self: typing.Self) -> str:
        return f"{self.feature} {self.op} {self.value}"


@dataclass(frozen=True)
class Rule:
    """TODO"""

    clauses: typing.Iterable[Clause | typing.Self]
    reducer: str

    training_data: pd.DataFrame = field(compare=False, hash=False, default_factory=pd.DataFrame)
    training_class: str | None = field(compare=False, hash=False, default=None)

    @property
    def score(self: typing.Self) -> ConfusionMatrix:
        """The rule score"""
        return compute_rule_score(self, self.training_data, class_attr=self.training_class)

    def evaluate(
            self: typing.Self,
            data: pd.DataFrame,
            *,
            add_eval_to_original_dataframe: bool = False,
    ) -> pd.Series:
        """Evaluate the rule against the given pd.DataFrame, returning a pd.Series with the result of the rule for each row"""
        # create a dataframe for the results of the clauses evaluations
        result = pd.DataFrame()

        # for each clause add a column to the dataframe
        for clause in self.clauses:
            result[clause.__repr__()] = clause.evaluate(
                data,
                add_eval_to_original_dataframe=add_eval_to_original_dataframe,
            )

        # compute the result of aggregating the results of the clauses for each row from the dataframe
        evaluation = _operators[self.reducer](result)

        # set the rule string representation as the series name
        evaluation = evaluation.rename(self.__repr__())

        # if asked, add the result of the evaluation to the input dataframe
        if add_eval_to_original_dataframe:
            data[self.__repr__()] = evaluation

        # return the result of evaluating the rule over the dataframe
        return evaluation

    def replace(self: typing.Self, pattern: str, replacement: str) -> typing.Self:
        """Replace the given pattern in the clauses attributes with the given replacement"""
        return Rule(
            clauses=[clause.replace(pattern, replacement) for clause in self.clauses],
            reducer=self.reducer,
            training_data=self.training_data,
            training_class=self.training_class,
        )

    def asdict(self: typing.Self) -> dict:
        return {
            "clauses": [clause.asdict() for clause in self.clauses],
            "reducer": self.reducer,
        }

    def __repr__(self: typing.Self) -> str:
        if len(list(self.clauses)) > 1:
            return f" {self.reducer} ".join([f"({clause.__repr__()})" for clause in self.clauses])

        return f" {self.reducer} ".join([f"{clause.__repr__()}" for clause in self.clauses])


def __encode_categorical_features(features: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder]:
    # store the names of the categorical columns
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns

    # create the OneHotEncoder instance.
    # prefer this approach over pd.get_dummies() because it allows us to revert the encoding later
    # also customize the format of the generated features to allow easier parsing when de-encoding
    encoder = OneHotEncoder(
        sparse_output=False,
        # if the category is nan, we set the column name to __DELETE__ to remove them later
        feature_name_combiner=lambda feature, category: f"[__{feature}__]: {category}" if not pd.isna(
            category) else "__DELETE__",
        # we want to encode the categories as booleans (the category is present or not)
        dtype=int,
    )

    # create a dataframe with the encoded features
    encoded_categorical = pd.DataFrame(
        # include only the categorical features
        encoder.fit_transform(features[categorical_columns]),
        # set the correct column names
        columns=encoder.get_feature_names_out(),
        # preserve the indices
        index=features.index,
    )

    # remove the columns where category is nan (if present)
    # we can remove them because a category of nan is just the lack of the rest of categories, and removing this column simplifies
    # things when decoding and simplifying the rules
    encoded_categorical = encoded_categorical.drop("__DELETE__", axis=1, errors="ignore")

    # concat both non-categorical and categorical features in a new dataframe
    encoded_features = pd.concat([features.drop(categorical_columns, axis=1), encoded_categorical], axis=1)

    # return the encoded data and the used encoder
    return encoded_features, encoder


def __decode_categorical_features(rule: Rule, encoder: OneHotEncoder) -> Rule:
    decoded_clauses = []

    # build a regex for parsing the artificial categorical feature names
    regex = r"\[__(?P<feature>.+)__\]: (?P<value>.+)"

    # parse each clause in the rule
    for clause in rule.clauses:
        # if the feature is encoded, try to parse it
        if clause.feature in encoder.get_feature_names_out():
            # parse the clause attr. the matching has to be complete. otherwise an error while parsing is thrown
            parsed = re.fullmatch(regex, clause.feature)

            # if the attr is not parsed correctly or the obtained feature name is not in the original categorical features,
            # an error is thrown as the feature can not be de-encoded
            if parsed is None or parsed.group("feature") not in encoder.feature_names_in_:
                raise Exception("Error parsing feature name " + clause.feature + " while decoding categorical features")

            # if everything went right, add the new, de-encoded clause to the decoded clauses list
            decoded_clauses.append(
                Clause(
                    # the original categorical feature
                    feature=parsed.group("feature"),
                    # the value obtained from the encoded categorical feature
                    value=parsed.group("value"),
                    # if the operator is >, then the feature should be true
                    op="==" if clause.op == ">" else "!=",
                ),
            )

        # otherwise, the feature was not categorical, so no transformation has to be done
        else:
            decoded_clauses.append(clause)

    return Rule(clauses=frozenset(decoded_clauses), reducer=rule.reducer, training_data=rule.training_data,
                training_class=rule.training_class)


def __simplify_categorical_clauses(rule: Rule, encoder: OneHotEncoder) -> Rule:
    # add the clauses involving non-categorical features to the simplified clauses list, as they will not be modified
    simplified_clauses = [clause for clause in rule.clauses if clause.feature not in encoder.feature_names_in_]

    # then, for each categorical feature in the encoder
    for feature in encoder.feature_names_in_:
        # get the clauses that have a condition over that feature
        clauses = [clause for clause in rule.clauses if clause.feature == feature]

        # find the positive and negative clauses for the given feature
        positive_clauses = [clause for clause in clauses if clause.op == "=="]
        negative_clauses = [clause for clause in clauses if clause.op == "!="]

        # get possible feature values, filtering np.nan from the possible values
        possible_values = {value for value in encoder.categories_[list(encoder.feature_names_in_).index(feature)] if
                           not pd.isna(value)}

        # if only one positive clause has been found for the feature, add it to the final list of clauses
        if len(positive_clauses) == 1:
            simplified_clauses.append(positive_clauses[0])
        # if no positive clause has been found, but all are negative and the count is (n of possible values -1), then
        # add the positive clause with the missing value
        elif len(positive_clauses) == 0 and len(negative_clauses) == (len(possible_values) - 1):
            # find the missing value
            missing_value = (possible_values - {clause.value for clause in negative_clauses}).pop()
            # add a clause with the missing positive value and discard the negative ones
            simplified_clauses.append(
                Clause(
                    feature=feature,
                    value=missing_value,
                    op="==",
                ),
            )
        # if none of the above is true, (i.e. there are more than one positive clause or more than one value is missing)
        else:
            # add all the clauses without modifying anything
            simplified_clauses.extend(clauses)

    # return a new rule with the simplified clauses
    return Rule(clauses=frozenset(simplified_clauses), reducer=rule.reducer, training_class=rule.training_class,
                training_data=rule.training_data)


def __is_rule_redundant(rule: Rule, rules: list[Rule], data: pd.DataFrame) -> bool:
    # a rule is redundant if is already present in the rule set or if it covers the exact same observations
    return any(r == rule or pd.Series(r.evaluate(data) == rule.evaluate(data)).all() for r in rules)


def __balance_data(
        data: pd.DataFrame,
        outcome: pd.Series,
        *,
        k_neighbours: int = 3,
        max_observations_per_class: int = 100,
) -> tuple[pd.DataFrame, pd.Series]:
    # balances observations from imbalanced input dataframe by over-sampling the minority class via SMOTE-NC, which
    # supports both numerical and categorical features (or SMOTE if no categorical features are found)

    # down-sample classes with more than 100 observations
    if pd.Series(outcome.value_counts() > max_observations_per_class).any():
        # create the random under-sampler
        sampler = RandomUnderSampler(
            # specify we want at most 100 observations from each class
            sampling_strategy={
                key: min(count, max_observations_per_class) for key, count in outcome.value_counts().to_dict().items()
            },
            random_state=42,
        )
        data, outcome = sampler.fit_resample(data, outcome)

    # if the initial dataframe has less than (number of neighbours for the SMOTE + 1) observations from the minority
    # class, apply first a random over sampler and then apply the SMOTE
    if pd.Series(outcome.value_counts() < k_neighbours + 1).any():
        # create the random over-sampler
        sampler = RandomOverSampler(
            # specify we want at least (number of neighbours for the SMOTE + 1) observations for each class
            sampling_strategy={
                key: max(count, k_neighbours + 1) for key, count in outcome.value_counts().to_dict().items()
            },
            random_state=42,
        )
        data, outcome = sampler.fit_resample(data, outcome)

    sampler = SMOTEN(
        # if all features are categorical, use SMOTE-N
        k_neighbors=k_neighbours,
        random_state=42,
    ) if all(isinstance(column_type, pd.api.types.CategoricalDtype) for column_type in data.dtypes) else SMOTENC(
        # if there is a mix of categorical and numerical data, use SMOTE-NC
        k_neighbors=k_neighbours,
        categorical_features="auto",
        random_state=42,
    ) if any(isinstance(column_type, pd.api.types.CategoricalDtype) for column_type in data.dtypes) else SMOTE(
        # if only numerical data, use SMOTE
        k_neighbors=k_neighbours,
        random_state=42,
    )

    # return the new resampled dataframe
    return sampler.fit_resample(data, outcome)


def compute_rule_score(
        rule: Rule,
        data: pd.DataFrame,
        *,
        class_attr: str = "class",
        n_samples: int = 1,
        sample_size: float | int = 1.0,
        sample_with_replacement: bool = False,
        seed: int = 42,
) -> ConfusionMatrix | typing.Iterable[ConfusionMatrix]:
    """Evaluate the given rule over the dataframe and compute the resulting confusion matrix"""
    if n_samples > 1:
        return [
            compute_rule_score(
                rule,
                data,
                class_attr=class_attr,
                n_samples=1,
                sample_size=sample_size,
                sample_with_replacement=sample_with_replacement,
                seed=index,
            ) for index in range(n_samples)
        ]

    sample = data.sample(
        n=sample_size,
        replace=sample_with_replacement,
        axis=0,
        random_state=seed,
    ) if isinstance(sample_size, int) else data.sample(
        frac=sample_size,
        replace=sample_with_replacement,
        axis=0,
        random_state=seed,
    )
    # evaluate the rule over the dataframe and obtain the result
    evaluation = rule.evaluate(sample)
    # build and return the confusion matrix
    return ConfusionMatrix(
        # TP = real positives marked as positive (true marked as true)
        true_positives=len(sample[sample[class_attr] & evaluation]) if class_attr in sample else 0,
        # TN = real negatives marked as negative (false marked as false)
        true_negatives=len(sample[~sample[class_attr] & ~evaluation]) if class_attr in sample else 0,
        # FP = real negatives marked as positive (false marked as true)
        false_positives=len(sample[~sample[class_attr] & evaluation]) if class_attr in sample else 0,
        # FN = real positives marked as negatives (true marked as false)
        false_negatives=len(sample[sample[class_attr] & ~evaluation]) if class_attr in sample else 0,
    )


def discover_rules(
        features: pd.DataFrame,
        *,
        class_attr: str = "class",
        balance_data: bool = True,
        encode_categorical: bool = True,
        simplify_rules: bool = True,
        remove_redundant_rules: bool = True,
        min_rule_precision: float = 0.9,
        min_rule_recall: float = 0.01,
) -> typing.Iterable[Rule]:
    """
    Discover the combination of rules that lead to the positive outcome from the features passed as an argument.

    These rules are discovered fitting a tree ensemble to the features, and then extracting the rules from the trees.

    Parameters
    ----------
    * `features`:               *the dataframe containing the features from where the rules will be extracted*
    * `class_attr`:             *the name of the column from the dataframe containing the class*
    * `balance_data`:           *whether to balance the data received as input before discovering the rules or not*
    * `encode_categorical`:     *a flag indicating if the categorical features in the dataframe should be encoded using a one hot encoder*
    * `simplify_rules`:         *whether the discovered rules should be simplified and the encoding reversed or not*
    * `remove_redundant_rules`: *whether to remove rules that cover the same exact observations as a previous rule*
    * `min_rule_precision`:     *the precision threshold for considering a rule as valid (precision is the rate of TP vs TP+FP)*
    * `min_rule_recall`:        *the recall threshold for considering a rule as valid (recall is the rate of TP vs TP+FN)*

    Returns
    -------
    * a rule that describes the outcome from the given data
    """
    # if the provided features dataframe is empty, or only one class is present, no rules are extracted
    if len(features) == 0 or len(np.unique(features.loc[:, class_attr])) == 1:
        return frozenset()

    # keep only the rows where the values are different (we do not need "repeated" observations)
    features = features.drop_duplicates()

    # split the features passed in two datasets, observations and outcome
    observations = features.loc[:, features.columns != class_attr]
    outcome = features.loc[:, class_attr]

    encoder = None

    # # encode categorical features if asked to
    if encode_categorical:
        (observations, encoder) = __encode_categorical_features(observations)

    # balance training data if asked
    if balance_data:
        (observations, outcome) = __balance_data(observations, outcome)

    # drop columns containing NA from the encoded DataFrame (they cannot be used for training the model)
    observations = observations.dropna(axis=1)

    # create the model for the tree ensemble, applying bootstrapping to the sample selection.
    # also, prune the rules with less than given precision and recall
    model = SkopeRulesClassifier(
        # extracted rules must have a minimum precision of min_rule_precision
        precision_min=min_rule_precision,
        # extracted rules must have a minimum recall of min_rule_recall
        recall_min=min_rule_recall,
        # the maximum of estimators is set to the number of features in the observations
        n_estimators=observations.columns.size,
        # remove the max depth constraint for the trees
        max_depth=None,
        # no perform deduplication of rules
        max_depth_duplication=None,
        # bootstrap the samples (i.e., extract samples with replacement) when building the ensemble
        bootstrap=True,
        # set a seed for rng for reproducible results
        random_state=42,
    )

    # fit the model to the available observations in the data
    # specifying the feature names allows for generating more interpretable rules
    try:
        model.fit(
            observations,
            outcome,
            feature_names=observations.columns,
        )
    # A ValueError can be thrown by Dataframe.query if too many parameters are passed to some internal methods using numpy
    # (see https://github.com/numpy/numpy/issues/4398)
    except ValueError:
        # monkey-patch Dataframe.query method to force the use of python engine instead of numpy, avoiding the error
        original = pd.DataFrame.query
        pd.DataFrame.query = functools.partialmethod(pd.DataFrame.query, engine="python")
        # train the model with the patched version of the query method
        model.fit(
            observations,
            outcome,
            feature_names=observations.columns,
        )
        # restore the original behaviour
        pd.DataFrame.query = original

    # transform the obtained rules to a more interpretable format
    rules = []
    # each rule obtained by the model can contain a combination of multiple clauses that must be fulfilled
    for original_rule in model.rules_:
        clauses = []

        # each entry in the dictionary represents a clause that should be added to the rule
        for ((attr, op), value) in original_rule.agg_dict.items():
            # try casting the value to the correct type
            try:
                value = np.array([value], dtype=observations[attr].dtype)[0]
            # if type is int and the condition is > .5 an exception will throw, so we transform the value manually
            except ValueError:
                if pd.api.types.is_integer_dtype(observations[attr].dtype) and "." in value:
                    match op:
                        # if the operator is > or >=, then replace it by > and put in the value the previous integer
                        case ">" | ">=":
                            value = np.array([value.split(".")[0]], dtype=observations[attr].dtype)[0]
                            op = ">"
                        # if the operator is < or <=, then replace it by < and put in the value the next integer
                        case "<" | "<=":
                            value = np.array([value.split(".")[0]], dtype=observations[attr].dtype)[0] + 1
                            op = "<"

            # add the clause with the value cast to the correct type
            clauses.append(Clause(attr, op, value))

        # create the rule with the given clauses, combining them with an AND
        rule = Rule(clauses=frozenset(clauses), reducer="&&", training_class=class_attr, training_data=features)

        # decode the categorical features if previously encoded
        if encode_categorical:
            rule = __decode_categorical_features(rule, encoder)

        # simplify rules if needed, removing redundant clauses
        if simplify_rules and encode_categorical:
            rule = __simplify_categorical_clauses(rule, encoder)

        # add the rule to the rule set if no rules have been added, if it is not redundant or if asked to not check for redundancies
        if not remove_redundant_rules or not __is_rule_redundant(rule, rules, features):
            rules.append(rule)

    # return the extracted rule set
    return frozenset(rules)


def filter_log(rule: Rule) -> typing.Callable[[Log], Log]:
    """Filter a log applying the given rule"""

    def _filter(log: Log) -> Log:
        # build a dataframe from the log
        log_dataframe = pd.json_normalize([event.asdict() for event in log], sep=".")
        # apply the rule to the dataframe, and get the indices of the events that fulfill the conditions
        indices_to_keep = log_dataframe.loc[rule.evaluate(log_dataframe), :].index.to_numpy()
        # return the events from the log in the indices obtained before
        filtered = itemgetter(*indices_to_keep)(log) if len(indices_to_keep) > 0 else []
        return [filtered] if isinstance(filtered, Event) else filtered

    return _filter
