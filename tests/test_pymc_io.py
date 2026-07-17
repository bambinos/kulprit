import pytest
import pymc as pm

from kulprit.projection.pymc_io import (
    add_switches,
    compile_mllk,
    turn_off_terms,
    get_model_information,
    get_term_variables,
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

    def test_get_term_variables_missing_term(self, pymc_model):
        with pytest.raises(KeyError):
            get_term_variables(pymc_model, ["term_that_does_not_exist"])


def test_get_term_variables_excludes_nested_sigma_hyperprior():
    """A hierarchical (nested) prior on sigma should be fully excluded, not just its name."""

    term_name = "Days|Subject"
    with pm.Model() as model:
        sigma_tau = pm.HalfNormal(f"{term_name}_sigma_tau", sigma=1)
        sigma = pm.HalfNormal(f"{term_name}_sigma", sigma=sigma_tau)
        offset = pm.Normal(f"{term_name}_offset", mu=0, sigma=1, shape=3)
        pm.Deterministic(term_name, offset * sigma)
        pm.Normal("Intercept", mu=0, sigma=1)
        pm.Normal("obs", mu=0, sigma=1, observed=[0.0, 1.0, 2.0])

    term_to_vars = get_term_variables(model, [term_name])
    active_vars = term_to_vars[term_name]

    assert f"{term_name}_offset" in active_vars
    assert f"{term_name}_sigma" not in active_vars
    assert f"{term_name}_sigma_tau" not in active_vars
