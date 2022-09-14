import math
import numpy as np

import numba as nb


LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype="int64",
)


@nb.njit
def fast_factorial(n):  # pragma: no cover
    if n > 20:
        return math.gamma(n + 1)  # inexact but fast computation of the factorial
    return LOOKUP_TABLE[n]


@nb.njit
def combination(n, k):
    return fast_factorial(n) / (fast_factorial(k) * fast_factorial(n - k))


@nb.njit
def gaussian_log_pdf(y, mean, sigma):  # pragma: no cover
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mean) / sigma) ** 2


@nb.njit
def gaussian_neg_llk(points, mean, sigma):  # pragma: no cover
    llk = []
    for y, m in zip(points, mean):
        llk.append(gaussian_log_pdf(y, m, sigma))
    return -sum(llk)


@nb.njit
def binomial_log_pdf(y, prob, trials):  # pragma: no cover
    return np.log(combination(trials, y) * (prob**y) * ((1 - prob) ** (trials - y)))


@nb.njit
def binomial_neg_llk(points, probs, trials):  # pragma: no cover
    llk = []
    for y, p, t in zip(points, probs, trials):
        llk.append(binomial_log_pdf(y, p, t))
    return -sum(llk)


@nb.njit
def poisson_log_pdf(y, lam):  # pragma: no cover
    return np.log((lam**y) * np.exp(-lam) / fast_factorial(y))


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
