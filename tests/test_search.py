import bambi as bmb
import kulprit as kpt

import numpy as np
import pandas as pd

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
num_draws * num_chains
posterior = model.fit(draws=num_draws, chains=num_chains)
# build reference model object
proj = kpt.Projector(model, posterior)


def test_forward():
    with pytest.raises(NotImplementedError):
        proj.search()
