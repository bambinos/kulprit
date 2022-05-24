from kulprit.families.family import Family
from kulprit.projection.losses.kld import KullbackLeiblerLoss

import pytest

from tests import KulpritTest


class MyFamily:
    def __init__(self) -> None:
        self.name = "half-cauchy"


class HalfCauchyFamily:
    def __init__(self) -> None:
        # initialise Half-Cauchy family object
        self.family = MyFamily()


class TestLoss(KulpritTest):
    """Test the loss functions used in the procedure."""

    def test_gaussian_kl_init(self, ref_model):
        """Test that the correct KLD loss function is initialised."""

        family = Family(ref_model.data)
        loss = KullbackLeiblerLoss(family)
        assert loss.family_name == "gaussian"

    def test_gaussian_kl(self, draws, ref_model):
        """Test that KLD(P, P) == 0."""

        family = Family(ref_model.data)
        loss = KullbackLeiblerLoss(family)
        div = loss.forward(draws, draws)
        assert div == 0.0

    def test_gaussian_kl_shape(self, draws, ref_model):
        """Test the shape of the loss output for compatibility with PyTorch."""

        family = Family(ref_model.data)
        loss = KullbackLeiblerLoss(family)
        div = loss.forward(draws, draws)
        assert div.shape == ()

    def test_unimplemented_family(self):
        """Test the error raised when unimplemented family is used."""
        with pytest.raises(NotImplementedError):
            bad_family = HalfCauchyFamily()
            KullbackLeiblerLoss(bad_family)
