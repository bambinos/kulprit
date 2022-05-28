import torch
import bambi as bmb
import arviz as az

import kulprit as kpt
from kulprit import ReferenceModel
from kulprit.projection.architecture import GLMArchitecture

import pytest
import copy

from tests import KulpritTest


class TestProjector(KulpritTest):
    """Test projection methods in the procedure."""

    def test_idata_is_none(self, bambi_model):
        """Test that some inference data is automatically produced when None."""

        no_idata_ref_model = ReferenceModel(bambi_model)
        assert no_idata_ref_model.data.structure.num_draws is not None

    def test_no_intercept_error(self):
        """Test that an error is raised when no intercept is present."""

        # load baseball data
        df = bmb.load_data("batting").head(10)
        # build model without an intercept
        bad_model = bmb.Model("p(H, AB) ~ 0 + playerID", df, family="binomial")
        bad_idata = az.from_json("tests/data/binomial.json")

        with pytest.raises(UserWarning):
            # build a bad reference model object
            kpt.ReferenceModel(bad_model, bad_idata)

    def test_architecture_forward(self, ref_model):
        """Test that the architecture forward method works."""

        architecture = GLMArchitecture(ref_model.data.structure)
        y = architecture.forward(ref_model.data.structure.X)

        assert y.shape == (
            ref_model.data.structure.num_draws,
            ref_model.data.structure.num_obs,
        )

    def test_analytic_project(self, ref_model):
        """Test that the analytic projection method works."""

        # project the reference model to some parameter subset
        sub_model = ref_model.project(terms=["x"])

        assert sub_model.structure.X.shape == (ref_model.data.structure.num_obs, 2)
        assert sub_model.structure.num_terms == 2
        assert sub_model.structure.model_size == 1

    def test_gradient_project(self, ref_model):
        """Test that the gradient projection method works."""

        # project the reference model to some parameter subset
        sub_model = ref_model.project(terms=["x"], method="gradient")

        assert sub_model.structure.X.shape == (ref_model.data.structure.num_obs, 2)
        assert sub_model.structure.num_terms == 2
        assert sub_model.structure.model_size == 1

    def test_projected_idata_dims(self, ref_model, bambi_model_idata):
        """Test that the dimensions of the projected inference data are correct."""

        # extract dimensions of projected idata
        sub_model = ref_model.project(terms=ref_model.data.structure.term_names)
        sub_model_idata = sub_model.idata
        print(sub_model_idata)
        print(sub_model_idata.observed_data)

        num_chain = len(sub_model_idata.posterior.coords.get("chain"))
        num_draw = len(sub_model_idata.posterior.coords.get("draw"))
        num_obs = len(
            sub_model_idata.observed_data.coords.get(
                f"{ref_model.data.structure.response_name}_dim_0"
            )
        )
        disp_shape = sub_model_idata.posterior.get(
            f"{ref_model.data.structure.response_name}_sigma"
        ).shape

        # ensure the restricted idata object has the same dimensions as that of the
        # reference model
        assert num_chain == len(bambi_model_idata.posterior.coords.get("chain"))
        assert num_draw == len(bambi_model_idata.posterior.coords.get("draw"))
        assert num_obs == len(
            bambi_model_idata.observed_data.coords.get(
                f"{ref_model.data.structure.response_name}_dim_0"
            )
        )
        assert (
            disp_shape
            == bambi_model_idata.posterior.data_vars.get(
                f"{ref_model.data.structure.response_name}_sigma"
            ).shape
        )

    def test_reshaping(self, ref_model):
        """Test that the reshaping is correct."""

        # extract the parameters from the reference model
        theta = torch.from_numpy(
            ref_model.data.idata.posterior.stack(samples=("chain", "draw"))[
                ref_model.data.structure.term_names
            ]
            .to_array()
            .transpose(*("samples", "variable"))
            .values
        ).float()

        # extract dimensions of the reference model idata object
        num_chain = len(ref_model.data.idata.posterior.coords.get("chain"))
        num_draw = len(ref_model.data.idata.posterior.coords.get("draw"))
        num_terms = ref_model.data.structure.num_terms

        # reshape torch tensor back to desired dimensions
        reshaped = torch.reshape(theta, (num_chain, num_draw, num_terms))

        # achieve similarly shaped tensor using xarray transposition
        transposed = torch.from_numpy(
            ref_model.data.idata.posterior[ref_model.data.structure.term_names]
            .to_array()
            .transpose(*("chain", "draw", "variable"))
            .values
        ).float()

        # ensure that these two methods both behave well
        assert (reshaped == transposed).all()

    def test_project_one_term(self, ref_model):
        """Test that the projection method works for a single term."""

        # project the reference model to some parameter subset
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        submodel = ref_model_copy.project(terms=1)
        assert submodel.structure.num_terms == 2

    def test_project_too_many_terms(self, ref_model):
        """Test that the projection method raises an error for too many terms."""

        with pytest.raises(UserWarning):
            # project the reference model to some parameter subset
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.search()
            ref_model_copy.project(terms=10)

    def test_project_negative_terms(self, ref_model):
        """Test that the projection method raises an error for negative terms."""

        with pytest.raises(UserWarning):
            # project the reference model to some parameter subset
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.search()
            ref_model_copy.project(terms=-1)

    def test_project_wrong_term_names(self, ref_model):
        """Test that the projection method raises an error for wrong term names."""

        with pytest.raises(UserWarning):
            # project the reference model to some parameter subset
            ref_model.project(terms=["spam", "ham"])
