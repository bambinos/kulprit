Kullback-Leibler projections for Bayesian model selection
=========================================================

|Tests|
|Coverage|
|Black|
  
.. |Tests| image:: https://github.com/bambinos/kulprit/actions/workflows/test.yml/badge.svg
    :target: https://github.com/bambinos/kulprit

.. |Coverage| image:: https://codecov.io/gh/bambinos/kulprit/branch/main/graph/badge.svg?token=SLJIK2O4C5 
    :target: https://codecov.io/gh/bambinos/kulprit

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

Kulprit is under active development so use it with care. If you find any bugs or have any feature requests, please open an `issue <https://github.com/bambinos/kulprit/issues>`_ on GitHub.

Besides this documentation, we also recommend you to read `Advances in projection predictive inference <https://arxiv.org/abs/2306.15581>`_. The paper is not about Kulprit, but introduces the theory behind Kulprit and also provides some practical advice. You may also find this `guide <https://avehtari.github.io/modelselection/CV-FAQ.html>`_
on Cross-Validation and model selection is useful.

Installation
============

Kulprit requires a working Python interpreter (3.10+). We recommend installing Python and key numerical libraries using the `Anaconda Distribution <https://www.anaconda.com/products/individual#Downloads>`_, which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), Kulprit itself can be installed in one line using pip:

.. code-block:: bash

    pip install kulprit


By default Kulprit performs a forward search, if you want to use Lasso (L1 search) you need to install `scikit-learn` package. You can install it using pip:

.. code-block:: bash
    
    pip install kulprit[lasso]

Alternatively, if you want the bleeding edge version of the package you can install it from GitHub:

.. code-block:: bash

    pip install git+https://github.com/bambinos/kulprit.git


Dependencies
============

Kulprit is tested on Python 3.10+. Dependencies are listed in `pyproject.toml` and should all be installed by the Kulprit installer; no further action should be required.


Contributing
============

We welcome contributions from interested individuals or groups!
For information about contributing to Kulprit check out our instructions, policies, and guidelines `here <https://github.com/bambinos/kulprit/blob/main/CONTRIBUTING.md>`_.


Contributors
============

See the `GitHub contributor page <https://github.com/bambinos/kulprit/graphs/contributors>`_.


Citation
========

If you find Kulprit useful in your work, please cite the following paper:

.. code-block:: latex

    @misc{mclatchie2024,
        title={Advances in projection predictive inference}, 
        author={Yann McLatchie and S{\"o}lvi R{\"o}gnvaldsson and Frank Weber and Aki Vehtari},
        year={2024},
        eprint={2306.15581},
        archivePrefix={arXiv},
        primaryClass={stat.ME},
        url={https://arxiv.org/abs/2306.15581}, 
    }


Donations
============

If you want to support Kulprit financially, you can `make a donation <https://numfocus.org/donate-to-pymc>`_ to our sister project PyMC.


Contents
========

.. toctree::
   :maxdepth: 2

   examples/quick-start
   
   api_reference

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
