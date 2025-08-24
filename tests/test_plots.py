import copy
import pytest

from kulprit import plot_compare, plot_dist, plot_forest


class TestPlots:
    """Test the search method of the model selection procedure."""

    NUM_CHAINS, NUM_DRAWS = 2, 50

    def test_plot_comparison(self, ref_model):
        """Test the compare plotting method."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        plot_compare(ref_model_copy.compare(min_model_size=1))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"submodels": [0, 1], "include_reference": True},
            {"figure_kwargs": {"figsize": (4, 4)}},
        ],
    )
    def test_plot_dist(self, ref_model, kwargs):
        """Test the density plotting method."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        plot_dist(ref_model_copy, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"submodels": [0, 1], "include_reference": True},
            {"figure_kwargs": {"figsize": (4, 4)}},
        ],
    )
    def test_plot_forest(self, ref_model, kwargs):
        """Test the density plotting method."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        plot_forest(ref_model_copy, **kwargs)
