import copy
import pytest
import bambi as bmb
import numpy as np
from kulprit.projection.search_strategies import _first_non_zero_idx
from kulprit import ProjectionPredictive


class TestSearch:
    """Test the search method of the model selection procedure."""

    NUM_CHAINS, NUM_DRAWS = 2, 50

    def test_forward(self, ref_model):
        """Test that the search path is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        assert [submodel.size for submodel in ref_model_copy.submodels([0, 1, 2])] == [0, 1, 2]

    def test_l1(self, ref_model):
        """Test that L1 search gives expected result."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project(method="l1")
        assert [submodel.size for submodel in ref_model_copy.submodels([0, 1, 2])] == [0, 1, 2]

    def test_l1_utils(self):
        """Test that L1 utility methods return expected result."""
        arr = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
        assert _first_non_zero_idx(arr) == {0: 1, 1: 2, 2: np.inf}

    def test_l1_categorical_error(self):
        """Test that an error is raised when using L1 with a categorical model."""

        data = bmb.load_data("carclaims")[::50]
        model_cat = bmb.Model("claimcst0 ~ C(agecat) + gender + area", data, family="gaussian")
        fitted_cat = model_cat.fit(
            draws=100,
            tune=100,
            idata_kwargs={"log_likelihood": True},
        )

        with pytest.raises(NotImplementedError):
            ref_model = ProjectionPredictive(model=model_cat, idata=fitted_cat)
            ref_model.project(method="l1")

    def test_bad_search_method(self, ref_model):
        """Test that an error is raised when an invalid search method is used."""

        with pytest.raises(ValueError):
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.project(method="bad_method")

    def test_search_too_many_terms(self, ref_model):
        """Test than an error is raise when early_stop > than all available terms."""

        with pytest.warns(UserWarning):
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.project(early_stop=10)
