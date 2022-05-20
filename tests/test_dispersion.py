from kulprit.projection.dispersion import DispersionProjectorFactory

import torch

import pytest

from . import KulpritTest


class TestDispersion(KulpritTest):
    """Test the dispersion parameter projection methods."""

    def test_disp_projector_init(self, ref_model):
        """Test that the correct dispersion projector object is initialised."""

        # build dispersion projector from reference model
        disp_projector = DispersionProjectorFactory(ref_model.data)
        assert disp_projector.family == "gaussian"

    def test_gaussian_disp_proj_shape(self, ref_model, disp_proj_data):
        """Test that the shape of the projected dispersion parameter is correct."""

        # build dispersion projection object
        disp_projector = DispersionProjectorFactory(ref_model.data)

        # extract necessary data
        X_perp, theta_perp, sigma_ast = disp_proj_data

        # project dispersion parameter
        sigma_perp = disp_projector.forward(theta_perp, X_perp)
        # test equivalent shapes
        assert sigma_perp.shape == sigma_ast.shape
