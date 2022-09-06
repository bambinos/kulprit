# kulprit

_(Pronounced: kuːl.prɪt)_

[![PyPI](https://img.shields.io/pypi/v/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - License](https://img.shields.io/pypi/l/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)

[Getting Started](https://colab.research.google.com/github/yannmclatchie/kulprit/blob/main/docs/notebooks/quick-start.ipynb) | [Documentation](https://yannmclatchie.github.io/kulprit) | [Contributing](https://github.com/yannmclatchie/kulprit/blob/main/CONTRIBUTING.md)

---

Kullback-Leibler projections for Bayesian model selection in Python.

## Example workflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yannmclatchie/kulprit/blob/main/docs/notebooks/quick-start.ipynb)

```python
import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt

import arviz as az
import matplotlib.pyplot as plt

# define the data
x = np.random.normal(0, 0.25, 121)
y = np.random.normal(0, 1, 121)
z = np.random.normal(2 * x + 3, 0.1)
data = pd.DataFrame({"x": x, "y": y, "z": z})

# fit the reference model
model = bmb.Model("z ~ x + y", data, family="gaussian")
idata = model.fit()

# build reference model object and perform the search procedure
ref_model = kpt.ReferenceModel(model, idata)
ref_model.search()

# compare the projected posterior densities of the submodels
ax = ref_model.plot_densities();

# compare submodels found in the search by LOO-CV ELPD
cmp, ax = ref_model.loo_compare(plot=True);
cmp

# project the reference model onto a chosen submodel size
submodel = ref_model.project(1)

# visualise projected parameters
ax = az.plot_posterior(submodel.idata, var_names=submodel.model.term_names);
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
