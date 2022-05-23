"""Analytic solver module."""

import torch

from kulprit.data.data import ModelData
from kulprit.families.family import Family
from kulprit.projection.solvers import BaseSolver


class AnalyticSolver(BaseSolver):
    """Analytic projection optimisation solver."""

    def __init__(
        self,
        data: ModelData,
        family: Family,
        num_iters: int = None,
        learning_rate: float = None,
    ):
        # log reference model data and family
        self.data = data
        self.family = family

    def solve(self, submodel_structure: torch.tensor) -> tuple:
        """Perform projection by gradient descent.

        Args:
            submodel_structure (SubModelStructure): The structure of the submodel
                being projected onto

        Returns
            tuple: A tuple of the projection solution along with the final loss
                value of the gradient descent
        """

        # compute and return solution
        solution = self.family.solve_analytic(submodel_structure=submodel_structure)
        return solution
