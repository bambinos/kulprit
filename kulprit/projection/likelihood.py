# pylint: disable=consider-using-generator
import math
import numpy as np

import numba as nb


LOOKUP_TABLE = np.array(
    [
        0.0,
        0.0,
        0.69314718,
        1.79175947,
        3.17805383,
        4.78749174,
        6.57925121,
        8.52516136,
        10.6046029,
        12.80182748,
        15.10441257,
        17.50230785,
        19.9872145,
        22.55216385,
        25.19122118,
        27.89927138,
        30.67186011,
        33.50507345,
        36.39544521,
        39.33988419,
        42.33561646,
    ]
)


@nb.njit
def log_factorial(n):
    if n > 20:
        return math.lgamma(n + 1)  # inexact but fast computation of the factorial
    return LOOKUP_TABLE[n]


@nb.njit
def log_binom_coeff(n, k):
    return log_factorial(n) - log_factorial(k) + log_factorial(n - k)


@nb.njit
def gaussian_log_pdf(y, mean, sigma):
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mean) / sigma) ** 2


@nb.njit
def gaussian_neg_llk(points, mean, sigma):
    llk = 0
    for y, m in zip(points, mean):
        llk -= gaussian_log_pdf(y, m, sigma)
    return llk


@nb.njit
def binomial_log_pdf(y, prob, trials):
    if prob == 0 or prob == 1 or y > trials:
        return -np.inf
    else:
        return log_binom_coeff(trials, y) + y * np.log(prob) + (trials - y) * np.log(1 - prob)


@nb.njit
def binomial_neg_llk(points, probs, trials):
    llk = 0
    for y, p, t in zip(points, probs, trials):
        llk -= binomial_log_pdf(y, p, t)
    return llk


@nb.njit
def poisson_log_pdf(y, lam):
    if lam == 0:
        return -np.inf
    else:
        return y * np.log(lam) - lam - log_factorial(y)


@nb.njit
def poisson_neg_llk(points, lam):
    llk = 0
    for y, l in zip(points, lam):
        llk -= poisson_log_pdf(y, l)
    return llk


LIKELIHOODS = {
    "gaussian": gaussian_neg_llk,
    "binomial": binomial_neg_llk,
    "poisson": poisson_neg_llk,
}
