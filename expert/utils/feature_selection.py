import re
import typing

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV, GenericUnivariateSelect, SelectFromModel, VarianceThreshold, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from expert.process_model import Event, Log


def __build_features(
        log: Log,
        predictors_extractor: typing.Callable[[Event], typing.Mapping],
        class_extractor: typing.Callable[[Event], float],
) -> tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    # transform event log to a list of dicts
    data = [{**predictors_extractor(event), "outcome": class_extractor(event)} for event in log]

    # create a dataframe from the list of dicts generated and infer the types of the object columns
    features = pd.json_normalize(data).infer_objects()

    # store outcome column
    outcome = features.loc[:, "outcome"].dt.total_seconds()
    # store categorical columns
    categorical_features = features.select_dtypes(include=["object", "string", "category"])
    # store the numerical columns
    numerical_columns = features.loc[:, features.columns != "outcome"].select_dtypes(exclude=["object", "string", "category"])

    # ensure categorical columns have type=category
    categorical_features = categorical_features.astype("category")
    # encode categorical features using a OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=False,
        # set columns for category names with the format "[__feature__]: value"
        feature_name_combiner=lambda feature, category: f"[__{feature}__]: {category}",
        # we want to encode the categories as booleans (the category is present or not)
        dtype=bool,
    )
    categorical_features = pd.DataFrame(
        # include only the categorical features
        encoder.fit_transform(categorical_features),
        # set the correct column names
        columns=encoder.get_feature_names_out(),
        # preserve the indices
        index=features.index,
    )

    # concatenate both categorical and numerical features
    features = pd.concat([numerical_columns, categorical_features], axis=1)

    # return the normalized, encoded data
    return features, outcome, encoder


def __decode_feature_names(encoder: OneHotEncoder, features: typing.Iterable[str]) -> set[str]:
    out_features = []
    regex = r"\[__(?P<feature>.+)__\]: (?P<value>.+)"

    for feature in features:
        if feature in encoder.get_feature_names_out():
            parsed = re.fullmatch(regex, feature)
            out_features.append(parsed.group("feature"))
        else:
            out_features.append(feature)

    return set(out_features)


def from_model(
        threshold: str | float | None = None,
        max_features: int | typing.Callable | None = None,
        estimator: BaseEstimator = LassoCV(cv=5, random_state=42),
) -> typing.Callable[[pd.DataFrame, pd.Series], typing.Iterable[str]]:
    """Select features fitting a model and getting the significant features from it"""
    # create a function that configures feature selector using the given params
    def f(data: pd.DataFrame, outcome: pd.Series) -> typing.Iterable[str]:
        feature_selector = SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            max_features=max_features,
        )

        # train the feature selector
        feature_selector.fit(data, outcome)

        # return the names of the selected features
        return feature_selector.get_feature_names_out()
    # return the generated function
    return f


def rfe(
        cv: int = 5,
        estimator: BaseEstimator = LinearRegression(),
) -> typing.Callable[[pd.DataFrame, pd.Series], typing.Iterable[str]]:
    """Select features applying a recursive feature elimination for fitting a model and getting the best-performing model"""
    # create a function that configures feature selector using the given params
    def f(data: pd.DataFrame, outcome: pd.Series) -> typing.Iterable[str]:
        # select important features using a recursive elimination approach, with 5-fold cross validation and using a Decision Tree as the estimator
        feature_selector = RFECV(
            estimator=estimator,
            cv=cv,
        )

        # train the feature selector
        feature_selector.fit(data, outcome)

        return feature_selector.get_feature_names_out()

    # return the generated function
    return f


def univariate(
        score_function: typing.Callable = f_regression,
        mode: str = "fwe",
        param: str | float | int = 0.05,
) -> typing.Callable[[pd.DataFrame, pd.Series], typing.Iterable[str]]:
    """Filter significant features based on univariate correlations"""
    def f(data: pd.DataFrame, outcome: pd.Series) -> typing.Iterable[str]:
        feature_selector = GenericUnivariateSelect(
            score_function,
            mode=mode,
            param=param,
        )
        # train the feature selector
        feature_selector.fit(data, outcome)

        return feature_selector.get_feature_names_out()
    return f


def chained_selectors(
        selectors: typing.Iterable[typing.Callable[[pd.DataFrame, pd.Series], typing.Iterable[str]]],
) -> typing.Callable[[pd.DataFrame, pd.Series], typing.Iterable[str]]:
    """Chain multiple selectors applying each one over the selected features for the previous"""
    # define the function that combines the selectors
    def f(data: pd.DataFrame, outcome: pd.Series) -> typing.Iterable[str]:
        # initially, all features are available
        features = data.columns.tolist()

        # for each selector, filter significant features from the available ones
        for selector in selectors:
            # update available features for the next iteration keeping only the ones selected by this selector
            features = selector(data.loc[:, features], outcome)

        # return the available features at the end, which are the relevant ones selected by the chain of selectors
        return features

    return f


def select_relevant_features(
        log: Log,
        *,
        predictors_extractor: typing.Callable[[Event], typing.Mapping],
        class_extractor: typing.Callable[[Event], float],
        feature_selector: typing.Callable[[pd.DataFrame, pd.Series], typing.Iterable[str]],
) -> tuple[set[str], pd.DataFrame]:
    """TODO"""
    # build the dataframe with the features from the log
    features, outcome, encoder = __build_features(log, predictors_extractor, class_extractor)

    # if no features available, just return
    if features.empty:
        return set(), features

    # normalize data (both predictors and outcome)
    scaler = MaxAbsScaler()
    features_scaled = pd.DataFrame(
        # include only the categorical features
        scaler.fit_transform(pd.concat([features, outcome], axis=1)),
        # set the correct column names
        columns=scaler.get_feature_names_out(),
    )
    # remove attributes with no variance
    try:
        variance_filter = VarianceThreshold(threshold=0.0)
        features_variance_filtered = pd.DataFrame(
            # include only the categorical features
            variance_filter.fit_transform(features_scaled),
            # set the correct column names
            columns=variance_filter.get_feature_names_out(),
        )
    # if none of the features has the minimum variance threshold, skip this step
    except ValueError:
        features_variance_filtered = features_scaled

    # ensure the outcome column is still present
    features_variance_filtered[outcome.name] = features_scaled[outcome.name]

    # select the important features and parse the name for the categorical encoded
    significant_feature_names = __decode_feature_names(
        encoder,
        feature_selector(
            features_variance_filtered.loc[:, features_variance_filtered.columns != outcome.name],
            features_variance_filtered.loc[:, outcome.name],
        ),
    )

    # return the dataframe representation of the log (reversing the one hot encoding) only with the significant features
    # decode features and convert back to categorical the corresponding columns
    decoded_categorical_features = pd.DataFrame(
        # include only the categorical features
        encoder.inverse_transform(features.loc[:, encoder.get_feature_names_out()]),
        # set the correct column names
        columns=encoder.feature_names_in_,
        # preserve the indices
        index=features.index,
    ).astype("category")

    # concat features with the decoded categorical columns and return the relevant features
    return significant_feature_names, pd.concat([features.drop(encoder.get_feature_names_out(), axis=1), decoded_categorical_features], axis=1)
