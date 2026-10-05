import numpy as np
import pytest
import pymc as pm

from kulprit.projection.pymc_io import (
    add_switches,
    compile_mllk,
    compile_marginal_mllk,
    turn_off_terms,
    get_model_information,
    get_term_partition,
    get_term_variables,
    ProbedMarginalObjective,
    _blocks_to_dense,
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
        assert len(var_info["x"]) == 4
        assert isinstance(var_info["x"][1], int)
        assert var_info["x"][2] is None
        assert var_info["x"][3] is None

        # transformed variables carry both directions and they round-trip
        backward, forward = var_info["sigma"][2], var_info["sigma"][3]
        assert backward is not None
        assert forward is not None
        target = 1.7
        unconstrained = float(np.ravel(forward(np.array([[target]])))[0])
        assert np.allclose(np.ravel(backward(np.array([[unconstrained]])))[0], target)

    def test_get_term_variables_missing_term(self, pymc_model):
        with pytest.raises(KeyError):
            get_term_variables(pymc_model, ["term_that_does_not_exist"])


def _hierarchical_model():
    """Small non-centered hierarchical model with a linked group-specific term."""

    term_name = "1|Group"
    group = np.array([0, 1, 2, 0, 1, 2])
    with pm.Model() as model:
        sigma = pm.HalfNormal(f"{term_name}_sigma", sigma=1)
        offset = pm.Normal(f"{term_name}_offset", mu=0, sigma=1, shape=3)
        coef = pm.Deterministic(term_name, offset * sigma)
        intercept = pm.Normal("Intercept", mu=0, sigma=1)
        pm.Normal(
            "obs",
            mu=intercept + coef[group],
            sigma=1,
            observed=np.array([0.5, -0.2, 1.1, 0.3, -0.4, 0.9]),
        )
    return model


def test_compile_marginal_mllk_without_group_terms(pymc_model):
    """Without group terms the marginal objective reduces to the plain likelihood."""

    term_names = [rv.name for rv in pymc_model.free_RVs]
    partition = get_term_partition(pymc_model, term_names)
    objective = compile_marginal_mllk(pymc_model, pymc_model.initial_point(), partition, {})

    initial_point = pymc_model.initial_point()
    params = np.concatenate([np.ravel(initial_point[key]) for key in initial_point])
    observed = pymc_model.rvs_to_values[pymc_model.observed_RVs[0]]
    y = np.zeros(np.asarray(observed.data).size)
    value, grad, hess = objective.dense(params, (y,))
    assert np.isfinite(value)
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()
    assert not objective.has_probe


def test_marginal_objective_counts_the_prior_once_for_pairs_of_rows():
    """A pair of pseudo-observation rows adds the observation terms but one prior."""

    model = _hierarchical_model()
    term_name = "1|Group"
    switched, switches = add_switches(model, [term_name])
    partition = get_term_partition(switched, [term_name])
    objective = compile_marginal_mllk(switched, switched.initial_point(), partition, switches)
    params, _ = _inner_offset_indices(switched)
    y0 = np.array([0.5, -0.2, 1.1, 0.3, -0.4, 0.9])
    y1 = np.array([-0.1, 0.4, 0.2, -0.7, 0.8, 0.1])

    pair_value = objective.dense(params, (y0, y1))[0]
    obs_only_y1 = objective._dense(params, y1, np.array(0.0))[0]  # pylint: disable=W0212
    assert pair_value == pytest.approx(objective.dense(params, (y0,))[0] + obs_only_y1)
    assert objective.dense(params, ((y0, y1),))[0] == pytest.approx(pair_value)


def _inner_offset_indices(model):
    initial_point = model.initial_point()
    flat_keys = [key for key in initial_point for _ in range(np.size(initial_point[key]))]
    params = np.concatenate([np.ravel(initial_point[key]) for key in initial_point])
    inner_idx = np.where(np.array(flat_keys) == "1|Group_offset")[0]
    return params, inner_idx


def _compiled_hierarchical_objective():
    model = _hierarchical_model()
    term_name = "1|Group"
    switched, switches = add_switches(model, [term_name])
    partition = get_term_partition(switched, [term_name])
    objective = compile_marginal_mllk(switched, switched.initial_point(), partition, switches)
    params, inner_idx = _inner_offset_indices(switched)
    return objective, params, inner_idx


def _assert_system_matches_dense(probed, objective, params, inner_idx, pred):
    value, grad, blocks = probed.system(params, pred)
    exact_value, exact_grad, exact_hess = objective.dense(params, pred)
    assert value == pytest.approx(float(exact_value))
    np.testing.assert_allclose(grad, exact_grad[inner_idx])
    np.testing.assert_allclose(
        _blocks_to_dense(blocks, inner_idx.size),
        exact_hess[np.ix_(inner_idx, inner_idx)],
        rtol=1e-8,
        atol=1e-8,
    )


def test_probed_marginal_objective_matches_autodiff_hessian(monkeypatch):
    """The Hessian-vector probe must reproduce the dense inner Hessian block."""

    monkeypatch.setattr(ProbedMarginalObjective, "MIN_INNER_FOR_PROBE", 0)
    objective, params, inner_idx = _compiled_hierarchical_objective()
    y0 = np.array([0.5, -0.2, 1.1, 0.3, -0.4, 0.9])
    y1 = np.array([-0.1, 0.4, 0.2, -0.7, 0.8, 0.1])

    probed = ProbedMarginalObjective(objective, inner_idx, params, y0, y1)
    assert probed.usable

    for pred in [(y0,), (y1,), (y0, y1)]:
        _assert_system_matches_dense(probed, objective, params, inner_idx, pred)


def test_probed_marginal_objective_skips_probe_for_few_inner_parameters():
    """Small inner sets use the dense autodiff Hessian and never compile the probe."""

    objective, params, inner_idx = _compiled_hierarchical_objective()
    y0 = np.array([0.5, -0.2, 1.1, 0.3, -0.4, 0.9])

    probed = ProbedMarginalObjective(objective, inner_idx, params, y0)
    assert not probed.usable
    _assert_system_matches_dense(probed, objective, params, inner_idx, (y0,))


def test_probed_marginal_objective_falls_back_without_probe(monkeypatch):
    """Without a compiled probe the wrapper must use the dense Hessian."""

    monkeypatch.setattr(ProbedMarginalObjective, "MIN_INNER_FOR_PROBE", 0)
    objective, params, inner_idx = _compiled_hierarchical_objective()
    objective._make_probe = None  # pylint: disable=protected-access
    y0 = np.array([0.5, -0.2, 1.1, 0.3, -0.4, 0.9])

    probed = ProbedMarginalObjective(objective, inner_idx, params, y0)
    assert not probed.usable
    _assert_system_matches_dense(probed, objective, params, inner_idx, (y0,))


def test_probed_marginal_objective_recovers_blocks_of_size_three(monkeypatch):
    """Three group-specific terms per group give blocks of size three (three probes)."""

    monkeypatch.setattr(ProbedMarginalObjective, "MIN_INNER_FOR_PROBE", 0)
    rng = np.random.default_rng(3)
    n_groups, n_obs = 4, 24
    group = np.tile(np.arange(n_groups), n_obs // n_groups)
    covariates = rng.normal(size=(2, n_obs))
    terms = ["1|G", "x|G", "z|G"]
    with pm.Model() as model:
        mu = 0.0
        for term, covariate in zip(terms, [np.ones(n_obs), *covariates]):
            sigma = pm.HalfNormal(f"{term}_sigma", sigma=1)
            offset = pm.Normal(f"{term}_offset", mu=0, sigma=1, shape=n_groups)
            mu = mu + pm.Deterministic(term, offset * sigma)[group] * covariate
        pm.Normal("obs", mu=mu, sigma=1, observed=rng.normal(size=n_obs))

    switched, switches = add_switches(model, terms)
    partition = get_term_partition(switched, terms)
    objective = compile_marginal_mllk(switched, switched.initial_point(), partition, switches)
    initial_point = switched.initial_point()
    keys = [key for key in initial_point for _ in range(np.size(initial_point[key]))]
    params = np.concatenate([np.ravel(initial_point[key]) for key in initial_point])
    inner_idx = np.where(np.char.endswith(np.array(keys), "_offset"))[0]
    y0, y1 = rng.normal(size=(2, n_obs))

    probed = ProbedMarginalObjective(objective, inner_idx, params, y0, y1)
    assert probed.usable
    assert probed._directions.shape[1] == 3  # pylint: disable=protected-access
    for pred in [(y0,), (y0, y1)]:
        _assert_system_matches_dense(probed, objective, params, inner_idx, pred)


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

    active_vars = get_term_variables(model, [term_name])[term_name]

    assert f"{term_name}_offset" in active_vars
    assert f"{term_name}_sigma" not in active_vars
    assert f"{term_name}_sigma_tau" not in active_vars
