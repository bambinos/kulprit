import numpy as np
import numba as nb


@nb.njit
def gaussian_log_pdf(y, mu, sigma):
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu) / sigma) ** 2


@nb.njit
def gaussian_neg_llk(points, mu, sigma):
    llk = []
    for y, m in zip(points, mu):
        llk.append(gaussian_log_pdf(y, m, sigma))
    return -sum(llk)


LIKELIHOODS = {
    "gaussian": gaussian_neg_llk,
}
