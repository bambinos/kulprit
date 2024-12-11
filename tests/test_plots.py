import copy
import pytest
import bambi as bmb

from kulprit import ProjectionPredictive


class TestPlots:
    """Test the search method of the model selection procedure."""

    NUM_CHAINS, NUM_DRAWS = 2, 50

    def test_plot_comparison(self, ref_model):
        """Test the compare plotting method."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        ref_model_copy.compare(plot=False, figsize=(10, 5), min_model_size=1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"kind": "forest"},
            {"kind": "forest", "plot_kwargs": {"combined": False}},
            {"submodels": [0, 1], "labels": "size"},
            {"figsize": (4, 4)},
        ],
    )
    def test_plot_densities(self, ref_model, kwargs):
        """Test the density plotting method."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        ref_model_copy.plot_densities(**kwargs)

    def test_loo(self, ref_model):
        """Test that the LOO score is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        elpd_info, _ = ref_model_copy.compare()
        assert [val[0] for val in elpd_info] == [-1, 0, 1, 2]

    def test_loo_with_no_search_path(self, ref_model):
        """Test that an error is raised when no search path is found."""

        with pytest.raises(UserWarning):
            data = bmb.load_data("my_data")
            bambi_model = bmb.Model("z ~ x + y", data, family="gaussian")
            idata = bambi_model.fit(draws=self.NUM_DRAWS, chains=self.NUM_CHAINS)
            ref_model = ProjectionPredictive(model=bambi_model, idata=idata)
            ref_model.compare()
