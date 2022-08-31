import numpy as np
import numba as nb


@nb.njit
def gaussian_neg_llk(points, mu, sigma):
    log_pdfs = np.array(
        [
            -1 * np.log(sigma)
            - 1 / 2 * np.log(2 * np.pi)
            - 1 / 2 * ((x - m) / sigma) ** 2
            for x, m in zip(points, mu)
        ]
    )
    llk = np.sum(log_pdfs)
    return -llk


LIKELIHOODS = {
    "gaussian": gaussian_neg_llk,
}
