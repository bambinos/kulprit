import math
import numpy as np
import numba as nb


@nb.njit
def gaussian_log_pdf(y, mean, sigma):  # pragma: no cover
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mean) / sigma) ** 2


@nb.njit
def gaussian_neg_llk(points, mean, sigma):  # pragma: no cover
    llk = []
    for y, m in zip(points, mean):
        llk.append(gaussian_log_pdf(y, m, sigma))
    return -sum(llk)


@nb.jit  # some error is being raised for no python implementation
def combination(n, k):  # pragma: no cover
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


@nb.njit
def binomial_log_pdf(y, mean, trials):  # pragma: no cover
    return np.log(combination(trials, y) * (mean**y) * ((1 - mean) ** (trials - y)))


@nb.njit
def binomial_neg_llk(points, mean, trials):  # pragma: no cover
    llk = []
    for y, m, t in zip(points, mean, trials):
        llk.append(binomial_log_pdf(y, m, t))
    return -sum(llk)


@nb.njit
def poisson_log_pdf(y, lam):  # pragma: no cover
    return np.log((lam**y) * np.exp(-lam) / math.factorial(y))


@nb.njit
def poisson_neg_llk(points, lam):  # pragma: no cover
    llk = []
    for y, l in zip(points, lam):
        llk.append(poisson_log_pdf(y, l))
    return -sum(llk)


LIKELIHOODS = {
    "gaussian": gaussian_neg_llk,
    "binomial": binomial_neg_llk,
    "poisson": poisson_neg_llk,
}
