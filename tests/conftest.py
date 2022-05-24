"""Module for storing fixtures used in kulprit tests."""

import kulprit as kpt
import bambi as bmb

import numpy as np

import torch

import pytest

# define model fitting options
NUM_DRAWS, NUM_CHAINS = 50, 2


@pytest.fixture(scope="session")
def bambi_model():
    # define model data
    data = data = bmb.load_data("my_data")
    # define and fit model with MCMC
    model = bmb.Model("z ~ x + y", data, family="gaussian")
    return model


@pytest.fixture(scope="session")
def bambi_model_idata(bambi_model):

    idata = bambi_model.fit(draws=NUM_DRAWS, chains=NUM_CHAINS)
    return idata


@pytest.fixture(scope="session")
def ref_model(bambi_model, bambi_model_idata):
    """Initialise a Gaussian reference model for use in later tests."""

    # build and return reference model object
    return kpt.ReferenceModel(bambi_model, bambi_model_idata)


@pytest.fixture(scope="session")
def sub_model(ref_model):
    """Initialise a standard submodel projection."""

    return ref_model.project(["x"])


@pytest.fixture(scope="session")
def disp_proj_data(ref_model, sub_model):
    """Produce data needed for dispersion projection tests."""

    # extract reference model data and parameters
    X_perp = sub_model.structure.X
    theta_perp = torch.from_numpy(
        sub_model.idata.posterior.stack(samples=("chain", "draw"))[
            sub_model.structure.term_names
        ]
        .to_array()
        .values.T
    ).float()

    # extract reference dispersion parameter
    sigma_ast = torch.from_numpy(
        ref_model.data.idata.posterior.stack(samples=("chain", "draw"))[
            ref_model.data.structure.response_name + "_sigma"
        ].values.T
    ).float()

    return X_perp, theta_perp, sigma_ast


@pytest.fixture(scope="session")
def draws():
    """Define some random Gaussian draw tensor to test loss methods."""

    return torch.from_numpy(np.random.normal(0, 1, 100)).float()
