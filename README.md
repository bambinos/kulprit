# kulprit

_(Pronounced: kuːl.prɪt)_

[![PyPI](https://img.shields.io/pypi/v/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - License](https://img.shields.io/pypi/l/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![Backend - PyTorch](https://img.shields.io/badge/backend-PyTorch-red?style=flat-square)](https://pytorch.org/)

[Getting Started](https://yannmclatchie.github.io/kulprit/examples) | [Documentation](https://yannmclatchie.github.io/kulprit) | [Contributing](https://github.com/yannmclatchie/kulprit/blob/main/CONTRIBUTING.md)

---

Kullback-Leibler projections for Bayesian model selection in Generalised Linear Models.

## Example workflow

🚧 **WIP** 🚧

```python
import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt

import arviz as az
import matplotlib.pyplot as plt

# define model data
data = pd.DataFrame({
    "y": np.random.normal(size=50),
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
# define and fit model with MCMC
model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
posterior = model.fit()
# build reference model object
proj = kpt.Projector(model, posterior)
# perform forward search model selection
model_search = proj.search(method="forward", max_terms=2)
# visualise search path
model_search.plot(stat="elpd")
# project the reference model to some parameter subset and plot posterior
cov_names = ["x1", "x2"]
theta_perp = proj.project(cov_names=cov_names)
az.plot_posterior(theta_perp)
plt.show()
```

## Installation

Currently, this package is only available for download directly from GitHub with the command
```bash
$ pip install git+https://github.com/yannmclatchie/kulprit.git
```

## Development

Read our development guide in [CONTRIBUTING.md](https://github.com/yannmclatchie/copenhagen/blob/master/CONTRIBUTING.md).

### Next steps

We are working on improving the package and workflow in the following ways:

1. Projecting the dispersion parameters of distributions (if applicable)
2. Adding a method to evaluate the ELPD of projected submodels
3. Implementing a forward search method
4. Checking the convergence of the projected parameter draws in terms of the rank-normalised R-hat of the projections
5. More distribution families implemented (binomial, poisson, etc.)
6. Some heuristic method to suggest the best submodel from a search procedure

---

This project was generated using the [biscuit](https://github.com/yannmclatchie/biscuit) cookiecutter.
