"""Module for storing fixtures used in kulprit tests."""

import kulprit as kpt
import bambi as bmb

import pytest

# define model fitting options
NUM_DRAWS, NUM_CHAINS = 50, 2


@pytest.fixture(scope="session")
def bambi_model():  # pragma: no cover
    """Return a bambi model."""

    # define model data
    data = bmb.load_data("my_data")
    # define model
    model = bmb.Model("z ~ x + y", data, family="gaussian")
    return model


@pytest.fixture(scope="session")
def bambi_model_idata(bambi_model):  # pragma: no cover
    """Return a bambi model fitted inference data."""

    # fit model with MCMC
    idata = bambi_model.fit(draws=NUM_DRAWS, chains=NUM_CHAINS)
    return idata


@pytest.fixture(scope="session")
def ref_model(bambi_model, bambi_model_idata):  # pragma: no cover
    """Initialise a Gaussian reference model for use in later tests."""

    # build and return reference model object
    return kpt.ReferenceModel(bambi_model, bambi_model_idata)


@pytest.fixture(scope="session")
def sub_model(ref_model):  # pragma: no cover
    """Initialise a standard submodel projection."""

    return ref_model.project(["x"])
