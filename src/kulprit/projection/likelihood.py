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
        51090942171709440000,
        1124000727777607680000,
    ],
    dtype="int64",
)


@nb.jit
def fast_factorial(n):  # pragma: no cover
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@nb.jit
def combination(n, k):  # pragma: no cover
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
