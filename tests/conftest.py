"""Module for storing fixtures used in kulprit tests."""

# pylint: disable=redefined-outer-name
import pytest

import bambi as bmb

import kulprit as kpt

# define model fitting options
NUM_DRAWS, NUM_CHAINS = 500, 4


@pytest.fixture(scope="session")
def bambi_model():  # pragma: no cover
    """Return a bambi model."""

    # define model data
    data = bmb.load_data("my_data")
    # define model
    model = bmb.Model("z ~ x + y", data, family="gaussian")
    return model


@pytest.fixture(scope="session")
def pymc_model(bambi_model):  # pragma: no cover
    """Return a bambi model."""

    bambi_model.build()
    model = bambi_model.backend.model
    return model


@pytest.fixture(scope="session")
def bambi_model_idata(bambi_model):  # pragma: no cover
    """Return a bambi model fitted inference data."""

    # fit model with MCMC
    idata = bambi_model.fit(
        draws=NUM_DRAWS, chains=NUM_CHAINS, idata_kwargs={"log_likelihood": True}
    )
    return idata


@pytest.fixture(scope="session")
def ref_model(bambi_model, bambi_model_idata):  # pragma: no cover
    """Initialise a Gaussian reference model for use in later tests."""

    # build and return reference model object
    return kpt.ProjectionPredictive(bambi_model, bambi_model_idata)


@pytest.fixture(scope="session")
def sub_model(ref_model):  # pragma: no cover
    """Initialise a standard submodel projection."""

    return ref_model.project(["x"])
