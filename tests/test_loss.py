from kulprit.families.family import Family
from kulprit.projection.losses import KullbackLeiblerLoss

import pytest

from tests import KulpritTest


class TestLoss(KulpritTest):
    """Test the loss functions used in the procedure."""

    def test_gaussian_kl(self, draws, ref_model):
        """Test that KLD(P, P) == 0."""

        loss = KullbackLeiblerLoss()
        div = loss.forward(draws, draws)
        assert div == pytest.approx(0.0)

    def test_gaussian_kl_shape(self, draws, ref_model):
        """Test the shape of the loss output for compatibility with PyTorch."""

        loss = KullbackLeiblerLoss()
        div = loss.forward(draws, draws)
        assert div.shape == ()
