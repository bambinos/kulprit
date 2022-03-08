import kulprit as kpt

import numpy as np
import torch

import pytest


def test_not_implemented_family_kl():
    with pytest.raises(NotImplementedError):
        kpt.families.Family.create("weibull")


def test_no_div_fun_family_kl():
    with pytest.raises(TypeError):

        class NewFamily(kpt.families.Family):
            _FAMILY_NAME = "my_new_family"

            def __init__(self):  # pragma: no cover
                super().__init__()

        torch.from_numpy(np.random.normal(0, 1, 100)).float()
        kpt.families.Family.create("my_new_family")


def test_gaussian_kl():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    family = kpt.families.Family.create("gaussian")
    assert family.kl_div(draws, draws) == 0.0
