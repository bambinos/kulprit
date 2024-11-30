from kulprit.projection.pymc_io import compile_mllk, compute_new_model, get_model_information
from tests import KulpritTest


class TestProjector(KulpritTest):
    """Test projection methods."""

    def test_compile_mllk(self, pymc_model):
        fmodel, old_y_value, obs_rvs = compile_mllk(pymc_model)
        assert callable(fmodel)
        assert old_y_value is not None
        assert obs_rvs is not None

    def test_compute_new_model(self, pymc_model):
        all_terms = [fvar.name for fvar in pymc_model.free_RVs]
        var_info = get_model_information(pymc_model)
        term_names = all_terms[-1:]
        new_model = compute_new_model(pymc_model, var_info, all_terms, term_names)
        assert new_model is not None

    def test_get_model_information(self, pymc_model):
        var_info = get_model_information(pymc_model)
        assert "x" in var_info
        assert isinstance(var_info["x"], tuple)
        assert len(var_info["x"]) == 3
        assert isinstance(var_info["x"][1], int)
        assert var_info["x"][2] is None
