import typing
from collections import Counter

import scipy

from expert.logger import LOGGER


def continuous_test(
        reference: typing.Iterable[float],
        running: typing.Iterable[float],
        *,
        alpha: float = 0.05,
) -> bool:
    """TODO DOCS"""
    if len(list(reference)) > 0 and len(list(running)) > 0:
        result = scipy.stats.mannwhitneyu(reference, running)

        LOGGER.verbose("test(reference != running) p-value: %.4f", result.pvalue)

        return result.pvalue < alpha

    return True


def categorical_test(
        reference: typing.Iterable[str],
        running: typing.Iterable[str],
        *,
        alpha: float = 0.05,
) -> bool:
    """TODO DOCS"""
    # get the set of possible values for the attribute, both in the reference and running datasets
    categories = {*reference, *running}
    # count values per category for each dataset
    reference_counts = Counter(reference)
    running_counts = Counter(running)
    # create a list with the frequencies of each possible category
    reference_frequencies = []
    running_frequencies = []
    # count the frequency of each category (we do not use the counter directly because it can be categories that are present
    # in one dataset but not the other, so the frequency would be missing
    for category in categories:
        reference_frequencies.append(reference_counts[category] if category in reference_counts else 0)
        running_frequencies.append(running_counts[category] if category in running_counts else 0)

    result = scipy.stats.chisquare(f_obs=reference_frequencies, f_exp=running_frequencies)

    LOGGER.verbose("test(reference != running) p-value: %.4f", result.pvalue)

    return result.pvalue < alpha
