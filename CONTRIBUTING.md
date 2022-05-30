# Contributing

Contributions to the package are very welcome! We recommend using `pyenv` to install a Python version compatible with `bambi` (these are versions `python>=3.7.2`), and then `poetry` for dependency management and virtual environment creation for development.

## Development

### Python version

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

### Dependency managament

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

Tests (whose scripts are located within the `kulprit/tests/` directory) can be run locally using `pytest` with the command
```sh
$ poetry run pytest
```

### Pre-commit checks

We use pre-commit hooks to automate formatting, linting, and quality checks. When developing locally, please initialise the pre-commit hooks with the command
```sh
$ poetry run pre-commit install
```
This will run the hooks with each commit. If you would like them to run only for each push:
```sh
$ pre-commit install -t pre-push
```
If you would like to run the hooks outwith a commit, then you can do so with
```sh
$ poetry run pre-commit run --all-files
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

One can also serve the docs locally by running
```bash
$ poetry run mkdocs serve
```
from the root directory.

## Releasing

Trigger the [Draft release workflow](https://github.com/yannmclatchie/kulprit/actions/workflows/draft_release.yml) (press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the [GitHub releases](https://github.com/yannmclatchie/kulprit/releases) and publish it. When a release is published, it'll trigger [release](https://github.com/yannmclatchie/kulprit/blob/main/.github/workflows/release.yml) workflow which creates PyPI release and deploys updated documentation.
