from kulprit.projection.pymc_io import (
    add_switches,
    compile_mllk,
    turn_off_terms,
    get_model_information,
)
from tests import KulpritTest


class TestProjector(KulpritTest):
    """Test projection methods."""

    def test_compile_mllk(self, pymc_model):
        neg_log_likelihood = compile_mllk(pymc_model, pymc_model.initial_point())
        assert callable(neg_log_likelihood)

    def test_compute_new_model(self, pymc_model):
        ref_terms = [fvar.name for fvar in pymc_model.free_RVs]
        _, switches = add_switches(pymc_model, ref_terms)
        term_names = ref_terms[-1:]
        turn_off_terms(switches, ref_terms, term_names)

    def test_get_model_information(self, pymc_model):
        var_info = get_model_information(pymc_model, pymc_model.initial_point())
        assert "x" in var_info
        assert isinstance(var_info["x"], tuple)
        assert len(var_info["x"]) == 3
        assert isinstance(var_info["x"][1], int)
        assert var_info["x"][2] is None
