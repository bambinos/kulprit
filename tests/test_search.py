import bambi as bmb
import kulprit as kpt

import copy
import pytest


class TestSearch:
    """Test the search method of the model selection procedure."""

    NUM_DRAWS, NUM_CHAINS = 50, 2

    def test_forward(self, ref_model):
        """Test that the search path is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        assert list(ref_model_copy.path.k_submodel.keys()) == [0, 1, 2]

    def test_search_too_many_terms(self, ref_model):
        with pytest.raises(UserWarning):
            ref_model_copy = copy.copy(ref_model)
            ref_model_copy.search()
            ref_model_copy.search(max_terms=10)

    def test_loo(self, ref_model):
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.search()
        all(ref_model_copy.loo_compare().index == [0, 1, 2])

    def test_loo_with_no_search_path(self, ref_model):
        with pytest.raises(UserWarning):
            # define model data
            data = bmb.load_data("my_data")
            # define and fit model with MCMC
            bambi_model = bmb.Model("z ~ x + y", data, family="gaussian")
            idata = bambi_model.fit(draws=self.NUM_DRAWS, chains=self.NUM_CHAINS)
            ref_model = kpt.ReferenceModel(model=bambi_model, idata=idata)
            ref_model.loo_compare()

    def test_repr(self, ref_model):
        ref_model_copy = copy.copy(ref_model)
        path = ref_model_copy.search()
        assert type(ref_model_copy.searcher.__repr__()) == str
        assert type(path.__repr__()) == str
