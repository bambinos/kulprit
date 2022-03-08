import kulprit as kpt
from kulprit.projection.divergences import KLDiv

import numpy as np

import torch

import pytest


def test_bad_family_kl():
    with pytest.raises(NotImplementedError):
        draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
        KLDiv.switch("weibull", draws, draws)


def test_gaussian_kl():
    draws = torch.from_numpy(np.random.normal(0, 1, 100)).float()
    kl_div = KLDiv.switch("gaussian", draws, draws)
    assert kl_div == 0.0
