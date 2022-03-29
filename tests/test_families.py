import kulprit as kpt
import bambi as bmb

import numpy as np
import pandas as pd

import torch

import pytest

import dataclasses

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
NUM_DRAWS, NUM_CHAINS = 100, 1
posterior = model.fit(draws=NUM_DRAWS, chains=NUM_CHAINS)
# build reference model object
proj = kpt.Projector(model, posterior)


def test_not_implemented_family():
    with pytest.raises(NotImplementedError):
        # load baseball data
        df = bmb.load_data("batting")
        # build model with a variate family not yet implemented
        bad_model = bmb.Model("p(H, AB) ~ 0 + playerID", df, family="binomial")
        # build reference model object
        kpt.Projector(bad_model, posterior)


def test_gaussian_kl():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    assert proj.ref_model.family.kl_div(draws, draws) == 0.0


def test_gaussian_disp_proj():
    # todo: extract theta_ast and theta_perp
    pass
    # build restricted model
    res_model = dataclasses.replace(proj.ref_model)
    proj.ref_model.family._project_disp_params(proj.ref_model, res_model)


def test_gaussian_disp_attribute():
    family = kpt.families.Family.create(model)
    assert family.has_disp_params


def test_gaussian_kl_shape():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    family = kpt.families.Family.create(model)
    div = family.kl_div(draws, draws)
    assert div.shape == ()
