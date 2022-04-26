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
    # build restricted model
    res_model = dataclasses.replace(proj.ref_model)

    # extract reference parameter draws and design matrix
    theta_perp = torch.from_numpy(
        res_model.idata.posterior.stack(samples=("chain", "draw"))[
            proj.ref_model.term_names
        ]
        .to_array()
        .values.T
    ).float()
    X_perp = res_model.X
    # perform projection of dispersion parameter
    sigma_perp = proj.ref_model.family._project_disp_params(
        proj.ref_model, theta_perp, X_perp
    )

    # extract reference dispersion parameter
    sigma_ast = torch.from_numpy(
        proj.ref_model.idata.posterior.stack(samples=("chain", "draw"))[
            proj.ref_model.response_name + "_sigma"
        ].values.T
    ).float()
    # test equivalent shapes
    assert sigma_perp.shape == sigma_ast.shape


def test_gaussian_disp_attribute():
    family = kpt.families.Family.create(model)
    assert family.has_disp_params


def test_gaussian_kl_shape():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    family = kpt.families.Family.create(model)
    div = family.kl_div(draws, draws)
    assert div.shape == ()
