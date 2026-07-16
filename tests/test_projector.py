import copy
import pytest

import numpy as np
import pandas as pd
import bambi as bmb

from kulprit import ProjectionPredictive
from kulprit.projector import _check_interactions
from kulprit.projection.solver import solve
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
            neg_log_likelihood=neg_log_likelihood,
            preds=[(np.array([0.0, 0.0]),)],
            initial_guess=np.array([0.0, 0.0]),
            var_info={"x": ((2,), 2, None)},
            weights=None,
            tolerance=1,
            active_idx=np.array([], dtype=int),
        )
