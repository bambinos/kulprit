import torch
from kulprit.families.family import Family
from kulprit.projection.losses.kld import KullbackLeiblerLoss, general_kl

import pytest

from tests import KulpritTest


class TestLoss(KulpritTest):
    """Test the loss functions used in the procedure."""

    def test_gaussian_kl(self, bambi_model):
        """Test that KLD(P, P) == 0 in the Gaussian case."""

        family = Family(model=bambi_model)
        loss = KullbackLeiblerLoss(family)
        linear_predictor, linear_predictor_ref = torch.normal(
            0, 1, size=(400,)
        ), torch.normal(0, 1, size=(400,))
        disp, disp_ref = torch.normal(1, 1, size=(400,)), torch.normal(1, 1, size=(400,))
        div = loss.forward(linear_predictor, disp, linear_predictor_ref, disp_ref)
        assert div == pytest.approx(0.0, abs=1e-1)
        assert div.shape == ()

    def test_general_kl(self, draws):
        """Test that KLD(P, P) == 0 in the general case."""

        div = general_kl(draws, draws)
        assert div == pytest.approx(0.0)
        assert div.shape == ()
