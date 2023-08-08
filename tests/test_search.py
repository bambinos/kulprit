# pylint: disable=no-self-use
import copy
import pytest

import bambi as bmb
import numpy as np

import kulprit as kpt
from kulprit.search.l1 import L1SearchPath


class TestSearch:
    """Test the search method of the model selection procedure."""

    NUM_DRAWS, NUM_CHAINS = 50, 2

    def test_forward(self, ref_model):
        """Test that the search path is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        assert list(ref_model_copy.path.keys()) == [0, 1, 2]

    def test_l1(self, ref_model):
        """Test that L1 search gives expected result."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search(method="l1")
        assert list(ref_model_copy.path.keys()) == [0, 1, 2]

    def test_l1_utils(self, ref_model):
        """Test that L1 utility methods return expected result."""

        proj = ref_model.projector
        searcher = L1SearchPath(proj)
        arr = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
        assert searcher.first_non_zero_idx(arr) == {0: 1, 1: 2, 2: np.inf}

    def test_l1_categorical_error(self):
        """Test that an error is raised when no search path is found."""

        data = bmb.load_data("carclaims")[::50]
        model_cat = bmb.Model("claimcst0 ~ C(agecat) + gender + area", data, family="gaussian")
        fitted_cat = model_cat.fit(
            draws=100,
            tune=100,
            idata_kwargs={"log_likelihood": True},
        )

        with pytest.raises(NotImplementedError):
            ref_model = kpt.ProjectionPredictive(model=model_cat, idata=fitted_cat)
            ref_model.search(method="l1")

    def test_bad_search_method(self, ref_model):
        """Test that an error is raised when an invalid search method is used."""

        with pytest.raises(UserWarning):
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.search(method="bad_method")

    def test_search_too_many_terms(self, ref_model):
        """Test than an error is raise when too many terms are used."""

        with pytest.raises(UserWarning):
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.search()
            ref_model_copy.search(max_terms=10)

    def test_loo(self, ref_model):
        """Test that the LOO score is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        cmp, _ = ref_model_copy.plot_compare()
        all(cmp.index == [0, 1, 2, 3])

    def test_loo_with_no_search_path(self, ref_model):
        """Test that an error is raised when no search path is found."""

        with pytest.raises(UserWarning):
            # define model data
            data = bmb.load_data("my_data")
            # define and fit model with MCMC
            bambi_model = bmb.Model("z ~ x + y", data, family="gaussian")
            idata = bambi_model.fit(draws=self.NUM_DRAWS, chains=self.NUM_CHAINS)
            ref_model = kpt.ProjectionPredictive(model=bambi_model, idata=idata)
            ref_model.plot_compare()

    def test_forward_repr(self, ref_model):
        """Test the string representation of the forward search path."""

        ref_model_copy = copy.copy(ref_model)
        path = ref_model_copy.search(method="forward")
        assert isinstance(ref_model_copy.searcher.__repr__(), str)
        assert isinstance(path.__repr__(), str)

    def test_l1_repr(self, ref_model):
        """Test the string representation of the L1 search path."""

        ref_model_copy = copy.copy(ref_model)
        path = ref_model_copy.search(method="l1")
        assert isinstance(ref_model_copy.searcher.__repr__(), str)
        assert isinstance(path.__repr__(), str)

    def test_plot_comparison(self, ref_model):
        """Test the LOO compare plotting method."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        ref_model_copy.plot_compare(plot=True, figsize=(10, 5))

    def test_plot_densities(self, ref_model):
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        ref_model_copy.plot_densities()
