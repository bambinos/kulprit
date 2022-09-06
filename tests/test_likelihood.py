import numpy as np

from scipy import stats

import pytest

from kulprit.projection.likelihood import gaussian_neg_llk


class TestLikelihood:
    """Test the likelihood mehtods implemented."""

    def test_gaussian_likelihood(self):
        # produce random samples
        data = list(np.random.random((10,)))

        # define the parameters of the Gaussian
        mus = list(np.random.random((10,)))
        sigma = 1.0

        # compute the log likelihood using scipy
        scipy_llk = np.log(np.product(stats.norm.pdf(data, mus, sigma)))

        # test that kulprit produces similar results
        assert -gaussian_neg_llk(data, mus, sigma) == pytest.approx(scipy_llk)
