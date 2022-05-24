import arviz as az
import bambi as bmb

import kulprit as kpt
from kulprit.families.family import Family
from kulprit.projection.losses.kld import KullbackLeiblerLoss

import pytest

from tests import KulpritTest


class TestFamily(KulpritTest):
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
