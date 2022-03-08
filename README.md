# kulprit

[![PyPI](https://img.shields.io/pypi/v/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![PyPI - License](https://img.shields.io/pypi/l/kulprit?style=flat-square)](https://pypi.python.org/pypi/kulprit/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://yannmclatchie.github.io/kulprit](https://yannmclatchie.github.io/kulprit)

**Source Code**: [https://github.com/yannmclatchie/kulprit](https://github.com/yannmclatchie/kulprit)

**PyPI**: [https://pypi.org/project/kulprit/](https://pypi.org/project/kulprit/)

---

Kullback-Leibler projections for Bayesian model selection in Generalised Linear Models.

## Example prototype workflow

Note that this proposed workflow is meant only to define the current iteration of the UI, and is by no means representative of this project's full scope.

```python
import pandas as pd
import numpy as np

import bambi as bmb
import kulprit as kpt

# define model data
data = pd.DataFrame({
    "y": np.random.normal(size=50),
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
# define and fit model with MCMC
model = bmb.Model("y ~ x1 + x2", data, family="gaussian")
num_draws, num_chains = 100, 1
posterior = model.fit(draws=num_draws, chains=num_chains)
# build reference model object
ref_model = kpt.Projector(model, posterior)
# project the reference model to some parameter subset
params = ["x1", "x2"]
theta_perp = ref_model.project(params=params)
print(theta_perp.shape)
# visualise the projected model posterior
ref_model.plot_projection(params=params)
```

## Installation

Currently, this package is only available for download directly from GitHub with the command
```bash
$ pip install git+https://github.com/yannmclatchie/kulprit.git
```

## Development

Contributions to the package are very welcome! We recommend using `pyenv` to install a Python version compatible with `bambi` (these are versions `python>=3.7.2`), and then `poetry` for dependency management and virtual environment creation for development.

For those using Mac, `pyenv` can be installed via homebrew with
```bash
$ brew install pyenv
```
and a new version of Python installed and applied in your local repo with
```bash
$ cd ~path/to/kulprit
$ pyenv install 3.7.2
$ pyenv local 3.7.2
$ eval "$(pyenv init --path)"
$ python --version # check that new version is being used
```

`poetry` can be installed with
```bash
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
and simply locking and installing the `pyproject.toml` file given in the repo with
```bash
$ cd ~path/to/kulprit
$ poetry config virtualenvs.in-project true
$ poetry env use $(which python)
$ poetry lock
$ poetry install
```
will spawn a virtual environment within the repo with all the necessary development tools and package requirements. Activate and work within this virtual environment by running
```bash
$ poetry shell
```

More information on the two tools can be found at the following links:
- [`poetry` documentation](https://python-poetry.org/)
- [`pyenv` documentation and repo](https://github.com/pyenv/pyenv)

### Testing

```sh
$ poetry run pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/yannmclatchie/kulprit/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/yannmclatchie/kulprit/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/yannmclatchie/kulprit/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
$ pre-commit install
```

Or if you want them to run only for each push:

```sh
$ pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
$ pre-commit run --all-files
```

## Project Organisation

```
    ├── LICENSE
    ├── Makefile               <- Makefile with commands like `make data` or `make train`
    ├── README.md              <- The top-level README for developers using this project
    │
    ├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
    |
    ├── mkdocs.yaml            <- Project configuration file
    │
    ├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `01-jqp-initial-data-exploration`
    │
    ├── poetry.lock            <- Poetry package management lock file
    ├── pyproject.toml         <- Poetry package management project dependency definition
    │
    ├── setup.cfg              <- Project configuration file
    ├── src
    |   └── kulprit            <- Source code for use in this project
    |       ├── __init__.py    <- Makes src a Python module
    |       ├── utils.py       <- Utility functions for workflow
    |       ├── plotting       <- Visualisation module
    |       |   ├── __init__.py
    |       |   └── visualise.py
    |       │
    |       ├── projection     <- Kullback-Leibler projections module
    |       |   ├── __init__.py
    |       |   ├── divergences.py
    |       |   ├── project.py
    |       |   └── submodel.py
    |       │
    |       └── search         <- Parameter search module
    |           ├── __init__.py
    |           └── forward.py
    │
    └── tests                  <- pytests module
        └── __init__.py
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
