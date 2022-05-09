import torch

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
idata = model.fit(draws=num_draws, chains=num_chains)
# build reference model object
proj = kpt.Projector(model, idata)


def test_idata_is_none():
    # test that some inference data has been automatically produced
    proj = kpt.Projector(model)

    assert proj.ref_model.num_draws is not None


def test_kl_opt_forward():
    solver = kpt.projection.optimise._KulOpt(proj.ref_model)
    y = solver.forward(proj.ref_model.X)

    assert y.shape == (proj.ref_model.num_draws, proj.ref_model.num_obs)


def test_project_method():
    # project the reference model to some parameter subset
    res_model = proj.project(model_size=2)

    assert res_model.X.shape == (proj.ref_model.num_obs, 3)
    assert res_model.num_terms == 3
    assert res_model.model_size == 2


def test_default_projection_set():
    # project the reference model to the default parameter subset
    res_model = proj.project()

    assert res_model.X.shape == proj.ref_model.X.shape
    assert res_model.num_terms == proj.ref_model.num_terms
    assert res_model.model_size == proj.ref_model.model_size


def test_zero_model_size_project():
    # project the reference model to zero term subset
    res_model = proj.project(model_size=0)

    assert res_model.X.shape == (proj.ref_model.num_obs, 1)
    assert res_model.num_terms == 1
    assert res_model.model_size == 0


def test_negative_model_size_project():
    with pytest.raises(UserWarning):
        # project the reference model to a negative model size
        proj.project(model_size=-1)


def test_too_large_model_size_project():
    with pytest.raises(UserWarning):
        # project the reference model to a parameter superset
        proj.project(model_size=proj.ref_model.num_terms + 1)


def test_projected_idata_dims():
    # extract dimensions of projected idata
    res_model = proj.project(model_size=0)
    idata_perp = res_model.idata
    num_chain = len(idata_perp.posterior.coords.get("chain"))
    num_draw = len(idata_perp.posterior.coords.get("draw"))
    num_obs = len(idata_perp.observed_data.coords.get("y_dim_0"))
    disp_shape = idata_perp.posterior.get("y_sigma").shape

    # ensure the restricted idata object has the same dimensions as that of the
    # reference model
    assert num_chain == len(idata.posterior.coords.get("chain"))
    assert num_draw == len(idata.posterior.coords.get("draw"))
    assert num_obs == len(idata.observed_data.coords.get("y_dim_0"))
    assert disp_shape == idata.posterior.data_vars.get("y_sigma").shape


def test_reshaping():
    """
    Ensure that torch reshaping is performing the true inverse of arviz stacking
    """

    # extract the parameters from the reference model
    ref_model = proj.ref_model
    theta = torch.from_numpy(
        ref_model.idata.posterior.stack(samples=("chain", "draw"))[ref_model.term_names]
        .to_array()
        .transpose(*("samples", "variable"))
        .values
    ).float()

    # extract dimensions of the reference model idata object
    num_chain = len(ref_model.idata.posterior.coords.get("chain"))
    num_draw = len(ref_model.idata.posterior.coords.get("draw"))
    num_terms = ref_model.num_terms

    # reshape torch tensor back to desired dimensions
    reshaped = torch.reshape(theta, (num_chain, num_draw, num_terms))

    # achieve similarly shaped tensor using xarray transposition
    transposed = torch.from_numpy(
        ref_model.idata.posterior[ref_model.term_names]
        .to_array()
        .transpose(*("chain", "draw", "variable"))
        .values
    ).float()

    # ensure that these two methods both behave well
    assert (reshaped == transposed).all()
