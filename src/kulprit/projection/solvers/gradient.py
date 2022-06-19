"""Gradient descent solver module."""

from typing import Optional

import torch

from kulprit.data.data import ModelData
from kulprit.families.family import Family
from kulprit.projection.architecture import Architecture
from kulprit.projection.losses import KullbackLeiblerLoss
from kulprit.projection.solvers import BaseSolver


class GradientDescentSolver(BaseSolver):
    """Gradient descent projection optimisation solver."""

    def __init__(
        self,
        data: ModelData,
        family: Family,
        num_iters: Optional[int] = 400,
        learning_rate: Optional[float] = 0.01,
    ):
        # log reference model data and family
        self.data = data
        self.family = family

        # log gradient descent parameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate

    def solve(self, submodel_structure: torch.tensor) -> tuple:
        """Perform projection by gradient descent.

        Args:
            submodel_structure (SubModelStructure): The structure of the submodel
                being projected onto

        Returns
            tuple: A tuple of the projection solution along with the final loss
                value of the gradient descent
        """

        # build architecture and loss methods for gradient descent
        self.architecture = Architecture(submodel_structure)
        self.loss = KullbackLeiblerLoss()

        # extract submodel design matrix
        X_perp = submodel_structure.X

        # extract thinned reference model posterior predictive samples
        y_ast = torch.from_numpy(
            self.data.structure.predictions.stack(samples=("chain", "draw"))
            .transpose(*("samples", f"{self.data.structure.response_name}_dim_0"))
            .values
        ).float()
        y_ast = y_ast[self.data.structure.thinned_idx]

        # project parameter samples and compute distance from reference model
        theta_perp, final_loss = self.optimise(X_perp, y_ast)
        return theta_perp, final_loss

    def optimise(self, X_perp, y_ast):
        """Optimisation loop in projection.

        Args:
            X_perp (torch.tensor):
            y_ast (torch.tensor):

        Returns:
            Tuple[torch.tensor, torch.tensor]: A tuple of the projected
                parameter draws as well as the final loss value (distance from
                reference model)
        """

        # build optimisation framework
        solver = self.architecture.architecture
        solver.zero_grad()
        optim = torch.optim.Adam(solver.parameters(), lr=self.learning_rate)

        # run optimisation loop
        for _ in range(self.num_iters):
            optim.zero_grad()
            y_perp = solver(X_perp)
            loss = self.loss.forward(y_perp, y_ast)
            loss.backward()
            optim.step()

        # extract projected parameters and final loss function value
        theta_perp = list(solver.parameters())[0].data
        final_loss = loss.item()
        return theta_perp, final_loss
