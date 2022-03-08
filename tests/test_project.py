import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt


# define model data
data = pd.DataFrame(
    {
        "y": np.random.normal(size=50),
        "g": np.random.choice(["Yes", "No"], size=50),
        "x1": np.random.normal(size=50),
        "x2": np.random.normal(size=50),
    }
)


def test_project_method():
    # define and fit model with MCMC
    model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
    num_draws, num_chains = 100, 1
    s = num_draws * num_chains
    posterior = model.fit(draws=num_draws, chains=num_chains)
    # build reference model object
    ref_model = kpt.Projector(model, posterior)
    # project the reference model to some parameter subset
    params = ["x1", "x2"]
    p = len(params) + 1
    theta_perp = ref_model.project(params=params)
    assert theta_perp.shape == (s, p)


def test_default_projection_set():
    # define and fit model with MCMC
    model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
    num_draws, num_chains = 100, 1
    s = num_draws * num_chains
    posterior = model.fit(draws=num_draws, chains=num_chains)
    # build reference model object
    ref_model = kpt.Projector(model, posterior)
    # project the reference model to some parameter subset
    theta_perp = ref_model.project()
    p = len(model.common_terms) + 1
    assert theta_perp.shape == (s, p)


def test_plot_projection():
    # define and fit model with MCMC
    model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
    num_draws, num_chains = 100, 1
    posterior = model.fit(draws=num_draws, chains=num_chains)
    # build reference model object
    ref_model = kpt.Projector(model, posterior)
    # project the reference model to some parameter subset
    params = ["x1", "x2"]
    ref_model.plot_projection(params=params)
