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
# define and fit model with MCMC
model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
num_draws, num_chains = 100, 1
posterior = model.fit(draws=num_draws, chains=num_chains)
# build reference model object
proj = kpt.Projector(model, posterior)


def test_posterior_is_none():
    kpt.Projector(model)


def test_kl_opt_forward():
    solver = kpt.projection.optimise._KulOpt(proj.ref_model)
    y = solver.forward(proj.ref_model.X)
    assert y.shape == (proj.ref_model.num_draws, proj.ref_model.num_obs)


def test_project_method():
    # project the reference model to some parameter subset
    proj.project(model_size=2)
    # to do: add shape test


def test_default_projection_set():
    # project the reference model to the default parameter subset
    proj.project()
    # to do: add shape test


def test_null_model_size_project():
    with pytest.raises(UserWarning):
        # project the reference model to the null parameter subset
        proj.project(model_size=0)


def test_too_large_model_size_project():
    with pytest.raises(UserWarning):
        # project the reference model to the null parameter subset
        proj.project(model_size=proj.ref_model.num_params + 1)


def test_elpd():
    with pytest.raises(NotImplementedError):
        kpt.utils._compute_elpd(proj.ref_model)
