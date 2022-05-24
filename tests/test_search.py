import pytest


class TestSearch:
    """Test the search method of the model selection procedure."""

    def test_forward(self, ref_model):
        with pytest.raises(NotImplementedError):
            ref_model.search()

    def test_search_too_many_terms(self, ref_model):
        with pytest.raises(UserWarning):
            ref_model.search(max_terms=10)
