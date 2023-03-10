import typing
import warnings

import scipy


def ks_test(alpha: float = 0.005) -> typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool]:
    """Compare the reference and running distributions using the KS test"""
    def __test(reference: typing.Iterable[float], running: typing.Iterable[float]) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = scipy.stats.ks_2samp(reference, running)

            return test.pvalue < alpha

    return __test


def cvm_test(alpha: float = 0.005) -> typing.Callable[[typing.Iterable[float], typing.Iterable[float]], bool]:
    """Compare the reference and running distributions using the Cramer von Misses test"""
    def __test(reference: typing.Iterable[float], running: typing.Iterable[float]) -> bool:
        return scipy.stats.cramervonmises_2samp(reference, running).pvalue < alpha
    return __test
