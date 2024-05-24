#
# import typing
#
# import numpy as np
# import pymc as pm
#
#
# # noinspection PyTypeChecker
# def probability_difference_under_threshold(
#         data1: typing.Iterable[float],
#         data2: typing.Iterable[float],
#         threshold: float = 0.5,
# ) -> float:
#     """Returns the probability of the difference of means being in the interval -threshold < diff < threshold"""
#     y1 = np.array(data1)
#     y2 = np.array(data2)
#     all_data = np.concatenate((y1, y2))
#     mu_m = np.mean(all_data)
#     mu_s = np.std(all_data) * 1000
#
#     sigma_low = mu_s / 1000
#     sigma_high = mu_s * 1000
#
#     nu_min = 2.5
#     nu_mean = 30
#     _nu_param = nu_mean - nu_min
#
#     with pm.Model():
#         group1_mean = pm.Normal("Group 1 mean", mu=mu_m, sigma=mu_s)
#         group2_mean = pm.Normal("Group 2 mean", mu=mu_m, sigma=mu_s)
#
#         nu = pm.Exponential(f"nu - {nu_min}", 1 / (nu_mean - nu_min)) + nu_min
#         _ = pm.Deterministic('Normality', nu)
#
#         group1_log_sigma = pm.Uniform(
#             'Group 1 log sigma', lower=np.log(sigma_low), upper=np.log(sigma_high),
#         )
#         group2_log_sigma = pm.Uniform(
#             'Group 2 log sigma', lower=np.log(sigma_low), upper=np.log(sigma_high),
#         )
#
#         group1_sigma = pm.Deterministic('Group 1 sigma', np.exp(group1_log_sigma))
#         group2_sigma = pm.Deterministic('Group 2 sigma', np.exp(group2_log_sigma))
#
#         lambda_1 = group1_sigma ** (-2)
#         lambda_2 = group2_sigma ** (-2)
#
#         group1_sd = pm.Deterministic('Group 1 SD', group1_sigma * (nu / (nu - 2)) ** 0.5)
#         group2_sd = pm.Deterministic('Group 2 SD', group2_sigma * (nu / (nu - 2)) ** 0.5)
#
#         _ = pm.StudentT("Group 1", nu=nu, mu=group1_mean, lam=lambda_1, observed=y1)
#         _ = pm.StudentT("Group 2", nu=nu, mu=group2_mean, lam=lambda_2, observed=y2)
#
#         diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
#         _ = pm.Deterministic('Difference of SDs', group1_sd - group2_sd)
#         _ = pm.Deterministic(
#             'Effect size', diff_of_means / np.sqrt((group1_sd ** 2 + group2_sd ** 2) / 2),
#         )
#
#         # idata = pm.sample(nuts_sampler="nutpie", random_seed=42, progressbar=True)
#         idata = pm.sample(nuts_sampler="numpyro", random_seed=42, progressbar=True)
#
#         samples = idata["posterior"]["Difference of means"].to_numpy().flatten()
#         n_match = len(samples[np.abs(samples) < abs(threshold)])
#         n_all = len(samples)
#
#         print(f"P(diff<{threshold}) = {n_match / n_all}")
#
#         return n_match / n_all
