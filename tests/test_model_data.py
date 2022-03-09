import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt
from kulprit.utils import _build_restricted_model


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
model_with_intercept = bmb.Model("y ~ x1 + x2", data, family="gaussian")
model_without_intercept = bmb.Model("y ~ -1 + x1 + x2", data, family="gaussian")
# define MCMC parameters
num_draws, num_chains = 100, 1
num_draws * num_chains
# fit the two models
posterior_intercept = model_with_intercept.fit(draws=num_draws, chains=num_chains)
posterior_no_intercept = model_without_intercept.fit(draws=num_draws, chains=num_chains)
# build two reference models
ref_model_intercept = kpt.Projector(model_with_intercept, posterior_intercept)
ref_model_no_intercept = kpt.Projector(model_without_intercept, posterior_no_intercept)


def test_has_intercept():
    assert ref_model_intercept.full_model.has_intercept


def test_does_not_have_intercept():
    assert not ref_model_no_intercept.full_model.has_intercept


def test_copy_reference_model():
    cov_names = ["x1", "x2"]
    res_model = _build_restricted_model(ref_model_intercept.full_model, cov_names)
    assert res_model.X.shape == (ref_model_intercept.full_model.n, len(cov_names) + 1)


def test_default_reference_model():
    res_model = _build_restricted_model(ref_model_intercept.full_model)
    assert res_model.X.shape == (
        ref_model_intercept.full_model.n,
        len(ref_model_intercept.full_model.cov_names) + 1,
    )


def test_build_restricted_model_with_intercept():
    cov_names = ["x1"]
    res_model = _build_restricted_model(ref_model_intercept.full_model, cov_names)
    assert res_model.X.shape == (ref_model_intercept.full_model.n, len(cov_names) + 1)


def test_build_restricted_model_without_intercept():
    cov_names = ["x1"]
    res_model = _build_restricted_model(ref_model_no_intercept.full_model, cov_names)
    assert res_model.X.shape == (ref_model_no_intercept.full_model.n, len(cov_names))
