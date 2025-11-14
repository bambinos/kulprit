<img src="https://raw.githubusercontent.com/bambinos/kulprit/main/docs/logos/kulprit_flat.png" width=200></img>

Kullback-Leibler projections for Bayesian model selection in Python.

[![PyPi version](https://badge.fury.io/py/kulprit.svg)](https://badge.fury.io/py/kulprit)
[![Build Status](https://github.com/bambinos/kulprit/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/kulprit/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/kulprit/branch/main/graph/badge.svg?token=SLJIK2O4C5)](https://codecov.io/gh/bambinos/kulprit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Overview

Kulprit _(Pronounced: kuːl.prɪt)_ is a package for variable selection for [Bambi](https://github.com/bambinos/bambi) models.
If you find any bugs or have any feature requests, please open an [issue](https://github.com/bambinos/kulprit/issues).


## Installation

Kulprit requires a working Python interpreter (3.11+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), Kulprit itself can be installed in one line using pip:

    pip install kulprit

By default, Kulprit performs a forward search. If you want to use Lasso (L1 search), you need to install `scikit-learn` package. You can install it using pip:

    pip install kulprit[lasso]

Alternatively, if you want the bleeding-edge version of the package, you can install it from GitHub:

    pip install git+https://github.com/bambinos/kulprit.git

## Documentation

The Kulprit documentation can be found in the [official docs](https://kulprit.readthedocs.io/en/latest/). The examples provide a quick overview of variable selection and how this problem is tackled by Kulprit. For a more detailed discussion of the theory, but also practical advice, we recommend the paper [Advances in Projection Predictive Inference](https://doi.org/10.1214/24-STS949).


## Contributions

Kulprit is a community project and welcomes contributions. Additional information can be found in the [CONTRIBUTING.md](https://github.com/bambinos/kulprit/blob/main/CONTRIBUTING.md) page.

For a list of contributors, see the [GitHub contributor](https://github.com/bambinos/kulprit/graphs/contributors) page

## Citation

If you use Kulprit and want to cite it, please use

```
@article{mclatchie2024,
    author = {Yann McLatchie and S{\"o}lvi R{\"o}gnvaldsson and Frank Weber and Aki Vehtari},
    title = {{Advances in Projection Predictive Inference}},
    volume = {40},
    journal = {Statistical Science},
    number = {1},
    publisher = {Institute of Mathematical Statistics},
    pages = {128 -- 147},
    keywords = {Bayesian model selection, cross-validation, projection predictive inference},
    year = {2025},
    doi = {10.1214/24-STS949},
    URL = {https://doi.org/10.1214/24-STS949}
}
```


## Donations

If you want to support Kulprit financially, you can [make a donation](https://numfocus.org/donate-to-pymc) to our sister project PyMC.

## Code of Conduct

Kulprit wishes to maintain a positive community. Additional details can be found in the [Code of Conduct](https://github.com/bambinos/kulprit/blob/main/docs/CODE_OF_CONDUCT.md)

## License

[MIT License](https://github.com/bambinos/kulprit/blob/main/LICENSE)
