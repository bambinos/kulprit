"""Continuous distribution families."""

from kulprit.data.data import ModelData
from kulprit.data.submodel import SubModelStructure
from kulprit.families import BaseFamily

import numpy as np
import torch


class GaussianFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = True
        self.has_analytic_solution = True
        self.name = "gaussian"

    def solve_analytic(self, submodel_structure: SubModelStructure) -> torch.tensor:
        """Analytic solution to the reference model parameter projection.

        Args:
            submodel_structure (SubModelStructure): The submodel structure object

        Returns
            tuple: A tuple of the projection solution along with the final loss
                value of the gradient descent
        """

        def _analytic_proj(theta_ast: np.float32) -> np.float32:
            """Analytic solution to the point-wise parameter projection.

            We separate this solution from the primary method to allow for
            vectorisation of the projection across samples.

            Args:
                theta_ast (np.float32): The reference model posterior
                    parameter samples

            Returns:
                np.float32: Analytic solution to the posterior parameter projection problem
            """

            f = X_ast @ theta_ast
            theta_perp = np.linalg.inv(X_perp.T @ X_perp) @ X_perp.T @ f
            return theta_perp

        # extract submodel design matrix
        X_perp = submodel_structure.X.numpy()

        # retrieve reference model design matrix
        X_ast = self.data.structure.X.numpy()

        # extract reference model posterior parameter samples
        theta_ast = (
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.term_names
            ]
            .to_array()
            .transpose(*("samples", "variable"))
            .values
        )

        # vectorise the analytic solution function
        vec_analytic_proj = np.vectorize(
            _analytic_proj,
            signature="(n)->(m)",
            doc="Vectorised `_analytic_proj` method",
        )

        # project the reference model posterior parameter samples
        theta_perp = torch.from_numpy(vec_analytic_proj(theta_ast)).float()
        theta_ast = torch.from_numpy(theta_ast).float()

        # compute the Kullback-Leibler divergence between projection and truth
        loss = None

        return theta_perp, loss

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic projection of the model dispersion parameters.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            torch.tensor: The restricted projections of the dispersion parameters
        """

        def _dispersion_proj(
            theta_ast: torch.tensor,
            theta_perp: torch.tensor,
            sigma_ast: torch.tensor,
        ) -> np.ndarray:
            """Analytic solution to the point-wise dispersion projection.

            We separate this solution from the primary method to allow for
            vectorisation of the projection across samples.

            Args:
                theta_ast (torch.tensor):
                theta_perp (torch.tensor):
                sigma_ast (torch.tensor):

            Returns:
                np.ndarray: The sample projection of the dispersion parameter in
                    a Gaussian model according to the analytic solution
            """

            f = self.X_ast @ theta_ast
            f_perp = self.X_perp @ theta_perp
            sigma_perp = torch.sqrt(
                sigma_ast**2
                + 1 / self.data.structure.num_obs * (f - f_perp).T @ (f - f_perp)
            )
            sigma_perp = sigma_perp.numpy()
            return sigma_perp

        # log the submodel design matrix
        self.X_perp = X_perp

        # extract parameter draws from both models
        theta_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.term_names
            ]
            .to_array()
            .transpose(*("samples", "variable"))
            .values
        ).float()
        sigma_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.response_name + "_sigma"
            ]
            .transpose()
            .values
        ).float()
        self.X_ast = self.data.structure.X

        # project the dispersion parameter
        vec_dispersion_proj = np.vectorize(
            _dispersion_proj,
            signature="(n),(m),()->()",
            doc="Vectorised `_dispersion_proj` method",
        )
        sigma_perp = (
            torch.from_numpy(vec_dispersion_proj(theta_ast, theta_perp, sigma_ast))
            .flatten()
            .float()
        )

        # assure correct shape
        assert sigma_perp.shape == sigma_ast.shape
        return sigma_perp
