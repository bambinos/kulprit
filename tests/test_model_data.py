import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt

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
idata = model.fit(draws=num_draws, chains=num_chains)
# build two reference models
proj = kpt.Projector(model, idata)


def test_has_intercept():
    assert proj.ref_model.has_intercept


def test_no_intercept_error():
    with pytest.raises(NotImplementedError):
        bad_model = bmb.Model("y ~ -1 + x1 + x2", data, family="gaussian")
        num_draws, num_chains = 100, 1
        bad_idata = bad_model.fit(draws=num_draws, chains=num_chains)
        kpt.Projector(bad_model, bad_idata)


def test_default_reference_model():
    res_model = proj._build_restricted_model()
    assert res_model.X.shape == (
        proj.ref_model.num_obs,
        proj.ref_model.num_terms,
    )


def test_build_restricted_model():
    model_size = 2
    res_model = proj._build_restricted_model(model_size=model_size)
    assert res_model.X.shape == (proj.ref_model.num_obs, model_size + 1)
    assert res_model.model_size == model_size


def test_build_negative_size_restricted_model():
    with pytest.raises(UserWarning):
        proj._build_restricted_model(model_size=-1)


def test_build_too_large_restricted_model():
    with pytest.raises(UserWarning):
        proj._build_restricted_model(model_size=100)
