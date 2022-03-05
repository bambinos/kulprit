# Projection predictive model selection

## Example prototype workflow

Note that this proposed workflow is meant only to define the next step of the desired UI, and is by no means representative of this project's full scope.

```python
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
posterior = model.fit()
# build reference model object
ref_model = kpt.Projector(model, posterior)
# project the reference model to `p` parameters
p = 1
sub_model = ref_model.project(ref_model, num_params=p)
# visualise the projected model posterior
sub_model.plot()
```

## Installation

Currently, this package is only available for download directly from GitHub with the command
```bash
pip install git+https://github.com/yannmclatchie/kulprit.git
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
will spawn a virtual environment within the repo with all the necessary development tools and package requirements.

More information on the two tools can be found at the following links:
- [`poetry` documentation](https://python-poetry.org/)
- [`pyenv` documentation and repo](https://github.com/pyenv/pyenv)

## Project Organization

```
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `01-jqp-initial-data-exploration`
    │
    ├── poetry.lock        <- Poetry package management lock file
    ├── pyproject.toml     <- Poetry package management project dependency definition
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── kulprit         <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── utils.py       <- Utility functions for workflow
    │   ├── plotting       <- Visualisation module
    │   |   └── visualise.py
    │   │
    |   ├── projection     <- Kullback-Leibler projections module
    │   |   ├── project.py
    │   |   └── divergences.py
    │   │
    |   └── search         <- Parameter search module
    │       └── forward.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

---
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
