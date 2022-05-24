from kulprit.families.family import Family

import pytest

from tests import KulpritTest


class TestDispersion(KulpritTest):
    """Test the dispersion parameter projection methods."""

    def test_gaussian_disp_proj_shape(self, ref_model, disp_proj_data):
        """Test that the shape of the projected dispersion parameter is correct."""

        # build dispersion projection object
        family = Family(ref_model.data)

        # extract necessary data
        X_perp, theta_perp, sigma_ast = disp_proj_data

        # project dispersion parameter
        sigma_perp = family.solve_dispersion(theta_perp, X_perp)
        # test equivalent shapes
        assert sigma_perp.shape == sigma_ast.shape
