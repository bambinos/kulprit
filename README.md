# kulprit

_(Pronounced: kuːl.prɪt)_

[![PyPI](https://img.shields.io/pypi/v/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - License](https://img.shields.io/pypi/l/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![Backend - PyTorch](https://img.shields.io/badge/backend-PyTorch-red?style=flat-square)](https://pytorch.org/)

[Getting Started](https://colab.research.google.com/github/yannmclatchie/kulprit/blob/main/docs/notebooks/quick-start.ipynb) | [Documentation](https://yannmclatchie.github.io/kulprit) | [Contributing](https://github.com/yannmclatchie/kulprit/blob/main/CONTRIBUTING.md)

---

Kullback-Leibler projections for Bayesian model selection in Generalised Linear Models.

## Example workflow

🚧 **WIP** 🚧

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yannmclatchie/kulprit/blob/main/docs/notebooks/quick-start.ipynb)

```python
import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt

import arviz as az
import matplotlib.pyplot as plt

# define model data
data = data = bmb.load_data("my_data")

# define and fit model with MCMC
model = bmb.Model("y ~ x + z", data, family="gaussian")
num_draws, num_chains = 2_000, 2
idata = model.fit(draws=num_draws, chains=num_chains)

# build reference model object and perform the search procedure
ref_model = kpt.ReferenceModel(model, idata)
ref_model.search()

# compare submodels found in the search
# setting option `plot=True` will return a comparison plot axes object
cmp, _ = ref_model.loo_compare()
cmp

# project the reference model onto a chosen submodel size
submodel = ref_model.project(1)

# visualise projected parameters
az.plot_posterior(submodel.idata)
```

## Installation

Currently, this package is only available for download directly from GitHub with the command
```bash
$ pip install git+https://github.com/yannmclatchie/kulprit.git
```

## Development

Read our development guide in [CONTRIBUTING.md](https://github.com/yannmclatchie/kulprit/blob/main/CONTRIBUTING.md).

---

This project was generated using the [biscuit](https://github.com/yannmclatchie/biscuit) cookiecutter.
