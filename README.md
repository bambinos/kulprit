# Projection predictive model selection

## Example prototype workflow

Note that this proposed workflow is meant only to define the next step of the desired UI, and is by no means representative of this project's full scope.

```python
import bambi as bmb
import pyprojpred as proj

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
ref_model = proj.Projector(model)
# project the reference model to `p` parameters
p = 1
sub_model = ref_model.project(ref_model, num_params=p)
# visualise the projected model posterior
sub_model.plot()
```

### Project Organization

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
    ├── pyprojpred         <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── plotting       <- Visualisation module
    │   |   └── visualise.py
    │   │
    |   ├── projection     <- Kullback-Leibler projections module
    │   │
    |   └── search         <- Parameter search module
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

---
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
