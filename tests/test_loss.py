import arviz as az
import bambi as bmb
import kulprit as kpt
from kulprit.projection.loss import KullbackLeiblerLoss

import torch

import pytest

from . import KulpritTest


class TestLoss(KulpritTest):
    """Test the loss functions used in the procedure."""

    NUM_DRAWS, NUM_CHAINS = 50, 2

    def test_not_implemented_family(self):
        """Test that unimplemented families raise a warning."""

        # load baseball data
        df = bmb.load_data("batting").head(50)
        # build model with a variate family not yet implemented
        bad_model = bmb.Model("p(H, AB) ~ 0 + playerID", df, family="binomial")
        bad_idata = az.from_json("tests/data/binomial.json")

        with pytest.raises(NotImplementedError):
            # build a bad reference model object
            kpt.ReferenceModel(bad_model, bad_idata)

    def test_gaussian_kl_init(self, ref_model):
        """Test that the correct KLD loss function is initialised."""

        loss = KullbackLeiblerLoss(ref_model.data)
        assert loss.family == "gaussian"

    def test_gaussian_kl(self, draws, ref_model):
        """Test that KLD(P, P) = 0."""

        loss = KullbackLeiblerLoss(ref_model.data)
        div = loss.forward(draws, draws)
        assert div == 0.0

    def test_gaussian_kl_shape(self, draws, ref_model):
        """Test the shape of the loss output for compatibility with PyTorch."""

        loss = KullbackLeiblerLoss(ref_model.data)
        div = loss.forward(draws, draws)
        assert div.shape == ()
