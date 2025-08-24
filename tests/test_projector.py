import copy
import pytest

import numpy as np
import pandas as pd
import bambi as bmb

from kulprit import ProjectionPredictive
from tests import KulpritTest


class TestProjector(KulpritTest):
    """Test projection methods."""

    NUM_CHAINS, NUM_DRAWS = 4, 500

    def test_idata_is_none(self, bambi_model):
        """Test that some inference data is automatically produced when None."""

        no_idata_ref_model = ProjectionPredictive(bambi_model)
        assert no_idata_ref_model.idata is not None

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
            ProjectionPredictive(bad_model, bambi_model_idata)

    def test_custom_path(self, bambi_model, bambi_model_idata):
        """Test user-defined path projection."""

        ppi = ProjectionPredictive(bambi_model, bambi_model_idata)
        # project the reference model to some parameter subset
        ppi.project(user_terms=[["x"]])

        sub_model_keys = ppi.list_of_submodels[0].idata.posterior.data_vars.keys()
        assert "x" in sub_model_keys
        assert "y" not in sub_model_keys

    def test_project_categorical(self):
        """Test that the projection method works with a categorical model."""

        data = bmb.load_data("carclaims")[::50]
        model_cat = bmb.Model("claimcst0 ~ C(agecat) + gender + area", data, family="gaussian")
        fitted_cat = model_cat.fit(
            draws=100,
            tune=100,
            idata_kwargs={"log_likelihood": True},
        )
        ppi = ProjectionPredictive(model=model_cat, idata=fitted_cat)
        ppi.project(user_terms=[["gender"]])
        assert ppi.list_of_submodels[0].size == 1

    def test_project_one_term(self, ref_model):
        """Test that the projection method works for a single term."""

        # project the reference model to some parameter subset
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        assert ref_model_copy.submodels(1).size == 1

    def test_loo(self, ref_model):
        """Test that the LOO score is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        cmp_df = ref_model_copy.compare()
        assert all(cmp_df.index == ["reference", "x", "y", "Intercept"])

    def test_loo_with_no_search_path(self, ref_model):
        """Test that an error is raised when no search path is found."""

        with pytest.raises(UserWarning):
            data = bmb.load_data("my_data")
            bambi_model = bmb.Model("z ~ x + y", data, family="gaussian")
            idata = bambi_model.fit(draws=self.NUM_DRAWS, chains=self.NUM_CHAINS)
            ref_model = ProjectionPredictive(model=bambi_model, idata=idata)
            ref_model.compare()
