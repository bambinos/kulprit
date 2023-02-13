import numpy as np

from scipy import stats

import pytest

from kulprit.projection.likelihood import (
    gaussian_neg_llk,
    binomial_neg_llk,
    poisson_neg_llk,
)


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

    def test_binomial_likelihood(self):
        # produce random samples
        data = list(np.random.randint(11, size=(10,)))

        # define the parameters of the Gaussian
        ns = list(np.random.randint(11, size=(10,)))
        probs = list(np.random.uniform(low=0, high=1, size=(10,)))
        print(type(ns), type(probs))
        print(ns, probs)

        # compute the log likelihood using scipy
        scipy_llk = np.log(np.product(stats.binom.pmf(data, ns, probs)))

        # test that kulprit produces similar results
        assert -binomial_neg_llk(data, ns, probs) == pytest.approx(scipy_llk)

    def test_poisson_likelihood(self):
        # produce random samples
        data = list(np.random.randint(11, size=(10,)))

        # define the parameters of the Gaussian
        lambdas = list(np.random.randint(11, size=(10,)))

        # compute the log likelihood using scipy
        scipy_llk = np.log(np.product(stats.poisson.pmf(data, lambdas)))

        # test that kulprit produces similar results
        assert -poisson_neg_llk(data, lambdas) == pytest.approx(scipy_llk)
