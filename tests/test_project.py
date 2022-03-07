import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt


def test_project_method():
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
    ref_model = kpt.Projector(model, posterior)
    # project the reference model to some parameter subset
    params = ["x1", "x2"]
    theta_perp = ref_model.project(params=params)  # noqa: F841


def test_default_projection_set():
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
    ref_model = kpt.Projector(model, posterior)
    # project the reference model to some parameter subset
    ref_model.project()


def test_plot_projection():
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
    ref_model = kpt.Projector(model, posterior)
    # project the reference model to some parameter subset
    params = ["x1", "x2"]
    ref_model.plot_projection(params=params)
