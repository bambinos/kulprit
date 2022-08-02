import numpy as np
import pandas as pd
import bambi as bmb

import kulprit as kpt
from kulprit import ReferenceModel

import pytest
import copy
from kulprit.projection.solver import Solver

from tests import KulpritTest
from tests.conftest import bambi_model_idata


class TestProjector(KulpritTest):
    """Test projection methods in the procedure."""

    NUM_DRAWS, NUM_CHAINS = 50, 2

    def test_idata_is_none(self, bambi_model):
        """Test that some inference data is automatically produced when None."""

        no_idata_ref_model = ReferenceModel(bambi_model)
        assert no_idata_ref_model.idata is not None

    def test_no_intercept_error(self):
        """Test that an error is raised when no intercept is present."""

        # define model data
        data = bmb.load_data("my_data")
        # define model
        bad_model = bmb.Model("z ~ 0 + x + y", data, family="gaussian")
        bad_idata = bad_model.fit(draws=self.NUM_DRAWS, chains=self.NUM_CHAINS)

        with pytest.raises(UserWarning):
            # build a bad reference model object
            kpt.ReferenceModel(bad_model, bad_idata)

    def test_incompatible_error(self, bambi_model_idata):
        """Test that an error is raised when model and idata aren't compatible."""

        # define model data
        a = np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
        b = np.array([59, 60, 62, 56, 63, 59, 62, 60])
        y = np.array([6, 13, 18, 28, 52, 53, 61, 60])
        data = pd.DataFrame({"a": a, "b": b, "y": y})

        # define model
        formula = "y ~ a + b"
        bad_model = bmb.Model(formula, data, family="gaussian")

        with pytest.raises(UserWarning):
            # build a bad reference model object
            kpt.ReferenceModel(bad_model, bambi_model_idata)

    def test_no_term_names_error(self, ref_model):
        """Test that an error is raised when no term names are provided."""

        with pytest.raises(UserWarning):
            # build a bad reference model object
            ref_model.project(terms=None)

    def test_project(self, ref_model):
        """Test that the analytic projection method works."""

        # project the reference model to some parameter subset
        sub_model = ref_model.project(terms=["x"])

        assert sub_model.num_chain == ref_model.idata.posterior.dims["chain"]
        assert sub_model.num_draw * sub_model.num_chain == 400

        response_name = list(ref_model.idata.observed_data.data_vars.keys())[0]
        assert (
            sub_model.num_obs
            == ref_model.idata.observed_data.dims[f"{response_name}_dim_0"]
        )
        assert sub_model.size == 1

    def test_project_categorical(self):
        """Test that the projection method works with a categorical model."""

        data = bmb.load_data("carclaims")
        data = data[data["claimcst0"] > 0]
        model_cat = bmb.Model(
            "claimcst0 ~ C(agecat) + gender + area", data, family="gaussian"
        )
        fitted_cat = model_cat.fit(draws=100, tune=2000, target_accept=0.9)
        ref_model = kpt.ReferenceModel(model=model_cat, idata=fitted_cat)
        sub_model = ref_model.project(terms=["gender"])
        assert sub_model.size == 1

    def test_project_one_term(self, ref_model):
        """Test that the projection method works for a single term."""

        # project the reference model to some parameter subset
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        submodel = ref_model_copy.project(terms=1)
        assert submodel.size == 1

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
