# pylint: disable=protected-access
import copy
import pytest

import numpy as np
import pandas as pd
import bambi as bmb

from kulprit import ProjectionPredictive
from kulprit.projector import _check_interactions
from kulprit.projection.solver import _blocks_logdet, _newton_inner, _solve_blocks, solve
from tests import KulpritTest


class TestProjector(KulpritTest):
    """Test projection methods."""

    NUM_CHAINS, NUM_DRAWS = 4, 500

    def test_different_variate_name(self, bambi_model_idata):
        """Test that an error is raised when model and idata aren't compatible."""

        # define model data
        data = pd.DataFrame(
            {
                "a": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
                "b": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
                "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
            }
        )

        # define model
        formula = "y ~ a + b"
        bad_model = bmb.Model(formula, data, family="gaussian")
        bad_model.build()

        with pytest.raises(UserWarning):
            # build a bad reference model object
            ProjectionPredictive(bad_model, bambi_model_idata)

    def test_custom_path(self, bambi_model, bambi_model_idata):
        """Test user-defined path projection."""

        ppi = ProjectionPredictive(bambi_model, bambi_model_idata)
        # project the reference model to some parameter subset
        ppi.project(user_terms=[["x"]])

        sub_model_keys = ppi[0].idata.posterior.data_vars.keys()
        assert "x" in sub_model_keys
        assert "y" not in sub_model_keys

    def test_project_categorical(self):
        """Test that the projection method works with a categorical model."""

        data = bmb.load_data("carclaims")[::50]
        model_cat = bmb.Model("claimcst0 ~ C(agecat) + gender + area", data, family="gaussian")
        fitted_cat = model_cat.fit(
            draws=100,
            tune=100,
            idata_kwargs={"log_likelihood": True},
        )
        ppi = ProjectionPredictive(model=model_cat, idata=fitted_cat)
        ppi.project(user_terms=[["gender"]])
        assert ppi[0].size == 1

    def test_project_one_term(self, ref_model):
        """Test that the projection method works for a single term."""

        # project the reference model to some parameter subset
        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        assert ref_model_copy[1].size == 1

    def test_loo(self, ref_model):
        """Test that the LOO score is as expected."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        cmp_df = ref_model_copy.compare()
        assert all(cmp_df.index == ["reference", "x", "y", "Intercept"])

    def test_compare_relative_to_reference(self, ref_model):
        """Test that compare with relative_to='reference' works correctly."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        cmp_df = ref_model_copy.compare(relative_to="reference")

        assert cmp_df.loc["reference", "elpd_diff"] == 0
        assert cmp_df.loc["reference", "dse"] == 0

        # Check that elpd_diff is computed correctly for submodels
        for idx in cmp_df.index:
            if idx != "reference":
                expected_diff = cmp_df.loc[idx, "elpd"] - cmp_df.loc["reference", "elpd"]
                assert abs(cmp_df.loc[idx, "elpd_diff"] - expected_diff) < 1e-10

    def test_compare_relative_to_full(self, ref_model):
        """Test that compare with relative_to='full' works correctly."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()
        cmp_df = ref_model_copy.compare(relative_to="full")

        # Find the full submodel (largest, last in the list)
        full_idx = cmp_df.index[1]  # First non-reference row is the full model

        # Full submodel should have 0 difference and 0 dse
        assert cmp_df.loc[full_idx, "elpd_diff"] == 0
        assert cmp_df.loc[full_idx, "dse"] == 0

        # Reference should have non-positive elpd_diff (worse or equal to full)
        assert cmp_df.loc["reference", "elpd_diff"] <= 0

        # Check that elpd_diff is computed correctly for all rows
        for idx in cmp_df.index:
            expected_diff = cmp_df.loc[idx, "elpd"] - cmp_df.loc[full_idx, "elpd"]
            assert abs(cmp_df.loc[idx, "elpd_diff"] - expected_diff) < 1e-10

    def test_select_relative_to_reference(self, ref_model):
        """Test that select with relative_to='reference' uses correct base."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()

        ref_elpd = ref_model_copy.reference_model.elpd
        selected = ref_model_copy.select(criterion="mean", relative_to="reference")
        assert (ref_elpd - selected.elpd) < 4

        for submodel in ref_model_copy._list_of_submodels:
            if submodel.size < selected.size:
                assert (ref_elpd - submodel.elpd) >= 4

    def test_select_relative_to_full(self, ref_model):
        """Test that select with relative_to='full' uses correct base."""

        ref_model_copy = copy.copy(ref_model)
        ref_model_copy.project()

        full_elpd = ref_model_copy._list_of_submodels[-1].elpd
        selected = ref_model_copy.select(criterion="mean", relative_to="full")

        assert (full_elpd - selected.elpd) < 4

        for submodel in ref_model_copy._list_of_submodels:
            if submodel.size < selected.size:
                assert (full_elpd - submodel.elpd) >= 4

    def test_loo_with_no_search_path(self, ref_model):
        """Test that an error is raised when no search path is found."""

        with pytest.raises(UserWarning):
            data = bmb.load_data("my_data")
            bambi_model = bmb.Model("z ~ x + y", data, family="gaussian")
            idata = bambi_model.fit(draws=self.NUM_DRAWS, chains=self.NUM_CHAINS)
            ref_model = ProjectionPredictive(model=bambi_model, idata=idata)
            ref_model.compare()

    def test_project_hierarchical_centered(self):
        """Test projection with centered group-specific terms."""

        data = bmb.load_data("sleepstudy")
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data, noncentered=False)
        idata = model.fit(draws=100, tune=100, chains=2, cores=1, random_seed=1234)
        model.compute_log_likelihood(idata)
        model.predict(idata, kind="response", random_seed=1234)

        ppi = ProjectionPredictive(model, idata)
        ppi.project(num_samples=30, num_clusters=5)

        elpd_values = [sub.elpd for sub in ppi]
        assert all(elpd_values[i] <= elpd_values[i + 1] for i in range(len(elpd_values) - 1))
        assert abs(ppi[-1].elpd - ppi.reference_model.elpd) < 20
        assert "Days|Subject_sigma" in ppi[-1].idata.posterior

    def test_project_hierarchical_non_centered(self):
        """Test projection with non-centered group-specific terms."""

        data = bmb.load_data("sleepstudy")
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data, noncentered=True)
        idata = model.fit(draws=100, tune=100, chains=2, cores=1, random_seed=1234)
        model.compute_log_likelihood(idata)
        model.predict(idata, kind="response", random_seed=1234)

        ppi = ProjectionPredictive(model, idata)
        ppi.project(num_samples=30, num_clusters=5)

        elpd_values = [sub.elpd for sub in ppi]
        assert all(elpd_values[i] <= elpd_values[i + 1] for i in range(len(elpd_values) - 1))
        assert abs(ppi[-1].elpd - ppi.reference_model.elpd) < 20

    def test_project_hierarchical_marginal_non_centered(self):
        """Marginal group-effect projection: per-draw sigma, ELPD parity, no side effects."""

        data = bmb.load_data("sleepstudy")
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data, noncentered=True)
        idata = model.fit(draws=100, tune=100, chains=2, cores=1, random_seed=1234)
        model.compute_log_likelihood(idata)
        model.predict(idata, kind="response", random_seed=1234)

        user_terms = [["Days"], ["Days", "Days|Subject"]]
        ppi = ProjectionPredictive(model, idata)
        ppi.project(user_terms=user_terms, num_samples=30)

        # sigma is re-estimated per projected draw near the reference scale
        # instead of being frozen at its initial value
        sigma = ppi[-1].idata.posterior["Days|Subject_sigma"].values
        reference_sigma = idata.posterior["Days|Subject_sigma"].values
        assert sigma.std() > 0
        assert 0.5 < sigma.mean() / reference_sigma.mean() < 2.0

        # submodels without group terms carry no group sigma
        assert "Days|Subject_sigma" not in ppi[0].idata.posterior
        assert "Days" in ppi[0].idata.posterior

        # predictive adequacy of the full projected model
        elpd_values = [sub.elpd for sub in ppi]
        assert all(elpd_values[i] <= elpd_values[i + 1] for i in range(len(elpd_values) - 1))
        assert abs(ppi[-1].elpd - ppi.reference_model.elpd) < 20

    def test_project_hierarchical_marginal_centered(self):
        """Marginal group-effect projection with a centered parameterization."""

        data = bmb.load_data("sleepstudy")
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data, noncentered=False)
        idata = model.fit(draws=100, tune=100, chains=2, cores=1, random_seed=1234)
        model.compute_log_likelihood(idata)
        model.predict(idata, kind="response", random_seed=1234)

        ppi = ProjectionPredictive(model, idata)
        ppi.project(user_terms=[["Days", "Days|Subject"]], num_samples=30)

        posterior = ppi[-1].idata.posterior
        sigma = posterior["Days|Subject_sigma"].values
        assert sigma.std() > 0
        assert "Days|Subject" in posterior
        assert abs(ppi[-1].elpd - ppi.reference_model.elpd) < 20


def test_check_interactions_group_specific_interaction_ok():
    """Grouped interactions are valid when fixed lower-order terms are present."""
    term_names = ["A", "B", "A:B", "A:B|Group"]
    _check_interactions(term_names, method="forward", require_lower_terms=True)


def test_check_interactions_group_specific_interaction_missing_fixed():
    """Grouped interactions raise when fixed-effect prerequisites are missing."""
    with pytest.raises(ValueError):
        _check_interactions(["A", "B", "A:B|Group"], method="forward", require_lower_terms=True)


def test_solve_raises_when_active_idx_empty():
    """The optimizer should fail fast when no parameters are active."""

    def neg_log_likelihood(params, pred):
        return np.sum((params - pred) ** 2)

    with pytest.raises(ValueError):
        solve(
            objective=neg_log_likelihood,
            preds=[(np.array([0.0, 0.0]),)],
            initial_guess=np.array([0.0, 0.0]),
            var_info={"x": ((2,), 2, None, None)},
            weights=None,
            tolerance=1,
            active_idx=np.array([], dtype=int),
        )


def _single_block(matrix):
    matrix = np.atleast_2d(matrix)
    return [(np.arange(len(matrix))[None], matrix[None])]


def test_blocks_logdet_rejects_indefinite_blocks():
    """Indefinite or non-finite blocks are unusable instead of floored."""

    assert _blocks_logdet(_single_block(np.diag([2.0, 3.0]))) == pytest.approx(np.log(6.0))
    assert _blocks_logdet(_single_block(np.diag([2.0, -1.0]))) is None
    assert _blocks_logdet(_single_block(np.array([[np.nan]]))) is None


def test_solve_blocks_matches_dense_solve():
    """Batched blocks of different sizes give the dense Newton step and log-determinant."""

    rng = np.random.default_rng(0)
    pairs = np.array([[0, 3], [1, 4]])
    triple = np.array([[2, 5, 6]])
    mats2 = np.array([[[2.0, 0.5], [0.5, 1.0]], [[3.0, -1.0], [-1.0, 2.0]]])
    raw = rng.normal(size=(3, 3))
    mats3 = (raw @ raw.T + 3 * np.eye(3))[None]
    dense = np.zeros((7, 7))
    for idx, mats in [(pairs, mats2), (triple, mats3)]:
        for ids, mat in zip(idx, mats):
            dense[np.ix_(ids, ids)] = mat
    grad = rng.normal(size=7)

    step, logdet = _solve_blocks([(pairs, mats2), (triple, mats3)], grad)
    np.testing.assert_allclose(step, -np.linalg.solve(dense, grad))
    assert logdet == pytest.approx(np.linalg.slogdet(dense)[1])


class _ScalarObjective:
    """One-dimensional inner objective defined by value, gradient and Hessian callables."""

    def __init__(self, value, grad, hess):
        self._value, self._grad, self._hess = value, grad, hess

    def system(self, params):
        z = params[0]
        return float(self._value(z)), np.array([self._grad(z)]), _single_block(self._hess(z))


def test_newton_inner_reaches_exact_mode():
    """The Newton inner solve converges to the exact mode of a non-quadratic objective."""

    objective = _ScalarObjective(lambda z: np.exp(z) - z, lambda z: np.exp(z) - 1, np.exp)
    z, value, logdet = _newton_inner(objective, np.array([3.0]), np.array([0]), ())
    np.testing.assert_allclose(z, [0.0], atol=1e-3)
    assert value == pytest.approx(1.0, abs=1e-6)
    assert logdet == pytest.approx(0.0, abs=1e-3)


def test_newton_inner_acts_on_inner_entries_only():
    """Outer entries of the full vector are held fixed while the inner ones are solved."""

    class Quadratic:
        def system(self, params):
            return 0.5 * float(np.sum(params**2)), params[:1].copy(), _single_block(np.eye(1))

    z, value, logdet = _newton_inner(Quadratic(), np.array([1.0, 2.0]), np.array([0]), ())
    np.testing.assert_allclose(z, [0.0], atol=1e-6)
    assert value == pytest.approx(2.0)
    assert logdet == 0.0


def test_newton_inner_handles_indefinite_hessian():
    """An indefinite Hessian is shifted away from the mode and reported unusable at one."""

    objective = _ScalarObjective(
        lambda z: 0.25 * z**4 - 0.5 * z**2, lambda z: z**3 - z, lambda z: 3 * z**2 - 1
    )
    # the stationary start is a local maximum: the block is unusable, not floored
    _, _, logdet = _newton_inner(objective, np.array([0.0]), np.array([0]), ())
    assert logdet is None

    # away from the saddle the shifted Newton steps reach a proper minimum
    z, _, logdet = _newton_inner(objective, np.array([0.5]), np.array([0]), ())
    assert abs(z[0]) == pytest.approx(1.0, abs=0.1)
    assert np.isfinite(logdet)
