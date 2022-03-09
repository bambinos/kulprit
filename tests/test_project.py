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
# define and fit model with MCMC
model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
num_draws, num_chains = 100, 1
num_draws * num_chains
posterior = model.fit(draws=num_draws, chains=num_chains)
# build reference model object
ref_model = kpt.Projector(model, posterior)


def test_kl_opt_forward():
    solver = kpt.projection.optimise._KulOpt(ref_model.full_model)
    y = solver.forward(ref_model.full_model.X)
    assert y.shape == (ref_model.full_model.s, ref_model.full_model.n)


def test_project_method():
    # project the reference model to some parameter subset
    cov_names = ["x1", "x2"]
    ref_model.project(cov_names=cov_names)
    # to do: add shape test


def test_default_projection_set():
    # project the reference model to the default parameter subset
    ref_model.project()
    # to do: add shape test


def test_plot_projection():
    # project the reference model to some parameter subset
    cov_names = ["x1", "x2"]
    ref_model.plot_projection(cov_names=cov_names)
    # to do: define more rigid test
