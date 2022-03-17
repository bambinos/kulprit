import kulprit as kpt
import bambi as bmb

import numpy as np
import pandas as pd

import torch

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


def test_not_implemented_family_kl():
    with pytest.raises(NotImplementedError):
        kpt.families.Family.create("weibull")


def test_no_div_fun_family_kl():
    with pytest.raises(TypeError):

        class NewFamily(kpt.families.Family):
            _FAMILY_NAME = "my_new_family"

            def __init__(self):  # pragma: no cover
                super().__init__()

        torch.from_numpy(np.random.normal(0, 1, 100)).float()
        kpt.families.Family.create("my_new_family")


def test_gaussian_kl():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    family = kpt.families.Family.create("gaussian")
    assert family.kl_div(draws, draws) == 0.0


def test_gaussian_disp_proj():
    # todo: extract theta_ast and theta_perp
    theta_ast = 1
    theta_perp = 2
    proj.full_model.family._project_disp_params(theta_ast, theta_perp)


def test_gaussian_disp_attribute():
    family = kpt.families.Family.create("gaussian")
    assert family.has_disp_params


def test_gaussian_kl_shape():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    family = kpt.families.Family.create("gaussian")
    div = family.kl_div(draws, draws)
    assert div.shape == ()
