# pylint: disable=:no-self-use
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
        data = np.random.random(10)

        # define the parameters of the Gaussian
        mus = np.random.random(10)
        sigma = 1.0

        # compute the log likelihood using scipy
        scipy_llk = stats.norm(mus, sigma).logpdf(data).sum()

        # test that kulprit produces similar results
        assert -gaussian_neg_llk(data, mus, sigma) == pytest.approx(scipy_llk)

    def test_binomial_likelihood(self):
        # produce random samples
        data = np.random.randint(11, size=10)

        # define the parameters of the Binomial
        ns_ = np.random.randint(11, size=10)
        probs = np.random.uniform(low=0, high=1, size=10)

        # compute the log likelihood using scipy
        scipy_llk = stats.binom(ns_, probs).logpmf(data).sum()

        # test that kulprit produces similar results
        assert -binomial_neg_llk(data, probs, ns_) == pytest.approx(scipy_llk)

    def test_poisson_likelihood(self):
        # produce random samples
        data = np.random.randint(11, size=10)

        # define the parameters of the Poisson
        lambdas = np.random.randint(11, size=10)

        # compute the log likelihood using scipy
        scipy_llk = stats.poisson(lambdas).logpmf(data).sum()

        # test that kulprit produces similar results
        assert -poisson_neg_llk(data, lambdas) == pytest.approx(scipy_llk)
