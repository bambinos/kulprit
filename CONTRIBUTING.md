# Contributing

Contributions to the package are very welcome! We recommend using `pyenv` to install a Python version compatible with `bambi` (these are versions `python>=3.7.2`), and then `poetry` for dependency management and virtual environment creation for development.

## Development

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

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

One can also serve the docs locally by running
```bash
$ poetry run mkdocs serve
```
from the root directory.

### Releasing

Trigger the [Draft release workflow](https://github.com/yannmclatchie/kulprit/actions/workflows/draft_release.yml) (press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the [GitHub releases](https://github.com/yannmclatchie/kulprit/releases) and publish it. When a release is published, it'll trigger [release](https://github.com/yannmclatchie/kulprit/blob/master/.github/workflows/release.yml) workflow which creates PyPI release and deploys updated documentation.

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
