import copy
import pytest

import numpy as np
import pandas as pd
import bambi as bmb

import kulprit as kpt
from kulprit import ProjectionPredictive
from tests import KulpritTest


class TestProjector(KulpritTest):
    """Test projection methods."""

    NUM_CHAINS, NUM_DRAWS = 4, 500

    def test_idata_is_none(self, bambi_model):
        """Test that some inference data is automatically produced when None."""

        no_idata_ref_model = ProjectionPredictive(bambi_model)
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
            kpt.ProjectionPredictive(bad_model, bad_idata)

    def test_hierarchical_error(self):
        """Test that an error is raised when model is hierarchical."""

        # define model data
        data = bmb.load_data("my_data")
        # define model
        bad_model = bmb.Model("z ~ (x|y)", data, family="gaussian")

        with pytest.raises(NotImplementedError):
            # build a bad reference model object
            kpt.ProjectionPredictive(bad_model)

    def test_different_variate_name(self, bambi_model_idata):
        """Test that an error is raised when model and idata aren't compatible."""

        # define model data
        data = pd.DataFrame(
            {
                "a": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
                "b": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
                "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
            }
        )

        # define model
        formula = "y ~ a + b"
        bad_model = bmb.Model(formula, data, family="gaussian")

        with pytest.raises(UserWarning):
            # build a bad reference model object
            kpt.ProjectionPredictive(bad_model, bambi_model_idata)

    def test_no_term_names_error(self, ref_model):
        """Test that an error is raised when no term names are provided."""

        with pytest.raises(UserWarning):
            # build a bad reference model object
            ref_model.project(terms=None)

    def test_projection(self, ref_model):
        """Test that the analytic projection method works."""

        # project the reference model to some parameter subset
        sub_model = ref_model.project(terms=["x"])

        sub_model_keys = sub_model.idata.posterior.data_vars.keys()
        assert "x" in sub_model_keys
        assert "y" not in sub_model_keys

    def test_project_categorical(self):
        """Test that the projection method works with a categorical model."""

        data = bmb.load_data("carclaims")[::50]
        model_cat = bmb.Model("claimcst0 ~ C(agecat) + gender + area", data, family="gaussian")
        fitted_cat = model_cat.fit(
            draws=100,
            tune=2000,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
        )
        ref_model = kpt.ProjectionPredictive(model=model_cat, idata=fitted_cat)
        sub_model = ref_model.project(terms=["gender"])
        assert sub_model.size == 1

    def test_project_one_term(self, ref_model):
        """Test that the projection method works for a single term."""

        # project the reference model to some parameter subset
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        submodel = ref_model_copy.project(terms=1)
        assert submodel.size == 1

    def test_project_tuple(self, ref_model):
        """Test that the projection method works for tuple input."""

        # project the reference model to some parameter subset
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        terms = tuple("x")
        submodel = ref_model_copy.project(terms=terms)
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
