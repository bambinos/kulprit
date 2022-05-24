from kulprit.data.data import ModelData
from kulprit.families import BaseFamily
from kulprit.families.family import Family
from kulprit.projection.solvers.solver import Solver

import pytest

from tests import KulpritTest


class TestSolver(KulpritTest):
    """Test the solver object methods."""

    def test_wrong_method(self, ref_model):
        """Test that the correct error is raised when a wrong method is used."""

        with pytest.raises(UserWarning):
            family = Family(ref_model.data)
            Solver(ref_model.data, family, method="random")
