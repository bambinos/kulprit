<img src="https://raw.githubusercontent.com/bambinos/kulprit/main/docs/logos/kulprit_flat.png" width=200></img>

Kullback-Leibler projections for Bayesian model selection in Python.

[![PyPi version](https://badge.fury.io/py/kulprit.svg)](https://badge.fury.io/py/kulprit)
[![Build Status](https://github.com/bambinos/kulprit/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/kulprit/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/kulprit/branch/main/graph/badge.svg?token=SLJIK2O4C5)](https://codecov.io/gh/bambinos/kulprit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Overview

Kulprit _(Pronounced: kuːl.prɪt)_ is a package for variable selection for [Bambi](https://github.com/bambinos/bambi) models.
Kulprit is under active development so use it with care. If you find any bugs or have any feature requests, please open an issue.


## Installation

Kulprit requires a working Python interpreter (3.9+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), Kulprit itself can be installed in one line using pip:

    pip install kulprit

Alternatively, if you want the bleeding edge version of the package you can install it from GitHub:

    pip install git+https://github.com/bambinos/kulprit.git

## Documentation

The Kulprit documentation can be found in the [official docs](https://kulprit.readthedocs.io/en/latest/). If you are not familiar with the theory behind Kulprit or need some practical advice on how to use Kulprit or interpret its results, we recommend you read the paper [Robust and efficient projection predictive inference](https://arxiv.org/abs/2306.15581). You may also find useful this [guide](https://avehtari.github.io/modelselection/CV-FAQ.html) on Cross-Validation and model selection.


## Development

Read our development guide in [CONTRIBUTING.md](https://github.com/bambinos/kulprit/blob/main/CONTRIBUTING.md).


## Contributions

Kulprit is a community project and welcomes contributions. Additional information can be found in the [Contributing](https://github.com/bambinos/kulprit/blob/main/docs/CONTRIBUTING.md) Readme.

For a list of contributors see the [GitHub contributor](https://github.com/bambinos/kulprit/graphs/contributors) page


## Citation

If you use Bambi and want to cite it please use

```
@misc{mclatchie2024,
      title={Advances in projection predictive inference}, 
      author={Yann McLatchie and Sölvi Rögnvaldsson and Frank Weber and Aki Vehtari},
      year={2024},
      eprint={2306.15581},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2306.15581}, 
}
```


## Donations

If you want to support Kulprit financially, you can [make a donation](https://numfocus.org/donate-to-pymc) to our sister project PyMC.

## Code of Conduct

Kulprit wishes to maintain a positive community. Additional details can be found in the [Code of Conduct](https://github.com/bambinos/kulprit/blob/main/docs/CODE_OF_CONDUCT.md)

## License

[MIT License](https://github.com/bambinos/kulprit/blob/main/LICENSE)
