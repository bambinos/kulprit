"""Base solver module."""

from typing import Optional
from typing_extensions import Literal

import torch

from kulprit.data.data import ModelData
from kulprit.data.submodel import SubModelStructure
from kulprit.families.family import Family
from kulprit.projection.solvers import BaseSolver
from kulprit.projection.solvers.analytic import AnalyticSolver
from kulprit.projection.solvers.gradient import GradientDescentSolver


class Solver:
    def __init__(
        self,
        data: ModelData,
        family: Family,
        method: Literal["analytic", "gradient"],
        num_iters: Optional[int] = 400,
        learning_rate: Optional[float] = 0.01,
    ) -> None:
        """Solver factory class.

        Args:
            data (ModelData): The reference model ModelData object
            method (str): The method of projection to use in the procedure
            num_iters (int): The number of iterations to run gradient descent
            learning_rate (float): The learning rate of the gradient descent
                optimiser
        """

        # log method, data, and family of the problem
        self.data = data
        self.method = method
        self.family = family

        # store gradient descent parameters if provided
        self.num_iters = num_iters
        self.learning_rate = learning_rate

        # define all available KL divergence loss classes
        self.method_dict = {
            "analytic": AnalyticSolver,
            "gradient": GradientDescentSolver,
        }

        # test valid solution method
        if self.method not in self.method_dict:
            raise UserWarning(
                "Please either solve the projection analytically or with "
                + "gradient descent."
            )

    def factory_method(self) -> BaseSolver:
        """Choose the appropriate family class given the model."""

        # return appropriate solver class given method
        solver_class = self.method_dict[self.method]
        return solver_class(
            data=self.data,
            family=self.family,
            num_iters=self.num_iters,
            learning_rate=self.learning_rate,
        )

    def solve(self, submodel_structure: SubModelStructure) -> tuple:
        """Solve the optimisation (projection) problem central to the procedure.

        Args:
            submodel_structure (SubModelStructure): The structure of the submodel
                being projected onto

        Returns
            tuple: A tuple of the projection solution along with the final loss
                value of the gradient descent
        """

        # initialise family class
        solver_class = self.factory_method()

        # compute the solution and return
        solution, loss = solver_class.solve(submodel_structure=submodel_structure)
        return solution, loss

    def solve_dispersion(
        self, theta_perp: torch.tensor, X_perp: torch.tensor
    ) -> torch.tensor:
        """Analytic projection of the model dispersion parameters.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            torch.tensor: The restricted projections of the dispersion parameters
        """

        # compute the solution and return
        solution = self.family.solve_dispersion(theta_perp=theta_perp, X_perp=X_perp)
        return solution
