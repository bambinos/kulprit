import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt
from kulprit.utils import _build_restricted_model

import pytest


# define model data
data = pd.DataFrame(
    {
        "y": np.random.normal(size=50),
        "g": np.random.choice(["Yes", "No"], size=50),
        "x1": np.random.normal(size=50),
        "x2": np.random.normal(size=50),
    }
)
# define two GLMs, one with intercept and one without
model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
# define MCMC parameters
num_draws, num_chains = 100, 1
# fit the two models
posterior = model.fit(draws=num_draws, chains=num_chains)
# build two reference models
proj = kpt.Projector(model, posterior)


def test_has_intercept():
    assert proj.ref_model.has_intercept


def test_no_intercept_error():
    with pytest.raises(NotImplementedError):
        bad_model = bmb.Model("y ~ -1 + x1 + x2", data, family="gaussian")
        num_draws, num_chains = 100, 1
        bad_posterior = bad_model.fit(draws=num_draws, chains=num_chains)
        kpt.Projector(bad_model, bad_posterior)


def test_copy_reference_model():
    cov_names = ["x1", "x2"]
    res_model = _build_restricted_model(proj.ref_model, cov_names)
    assert res_model.X.shape == (proj.ref_model.num_obs, len(cov_names) + 1)


def test_default_reference_model():
    res_model = _build_restricted_model(proj.ref_model)
    assert res_model.X.shape == (
        proj.ref_model.num_obs,
        len(proj.ref_model.cov_names) + 1,
    )


def test_build_restricted_model():
    cov_names = ["x1"]
    res_model = _build_restricted_model(proj.ref_model, cov_names)
    assert res_model.X.shape == (proj.ref_model.num_obs, len(cov_names) + 1)
