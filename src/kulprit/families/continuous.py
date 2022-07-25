"""Continuous distribution families."""

from bambi import Model
from kulprit.families import BaseFamily
from kulprit.families.link import Link

import numpy as np
import torch


class GaussianFamily(BaseFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def __init__(self, model: Model, link: Link) -> None:
        # initialise family object with necessary attributes
        self.model = model
        self.has_dispersion_parameters = True
        self.name = "gaussian"
        self.link = link

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
                theta_ast (torch.tensor): Reference model posterior parameter
                    sample
                theta_perp (torch.tensor): Submodel projected parameter sample
                sigma_ast (torch.tensor): Reference model posterior dispersion
                    parameter sample

            Returns:
                np.ndarray: The sample projection of the dispersion parameter in
                    a Gaussian model according to the analytic solution
            """

            f = X_ast @ theta_ast
            f_perp = X_perp @ theta_perp
            sigma_perp = torch.sqrt(
                sigma_ast**2
                + 1 / self.data.structure.num_obs * (f - f_perp).T @ (f - f_perp)
            )
            sigma_perp = sigma_perp.numpy()
            return sigma_perp

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

        # thin the parameter draws to match the optimisation
        theta_ast = theta_ast[self.data.structure.thinned_idx]
        sigma_ast = sigma_ast[self.data.structure.thinned_idx]

        # extract the design matrix of the reference model
        X_ast = self.data.structure.X

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

    def posterior_predictive(self, linear_predictor, disp):
        """Sample from posterior predictive distribution."""

        mean = self.link.linkinv(linear_predictor)
        return torch.normal(mean, disp)

    def kl_div(self, kwargs):
        """Kullback-Leibler divergence between two Gaussians.

        TODO:
            * Compute the analytic KL divergence between two Gaussians given the
                reference and restricted model parameters
        """

        raise NotImplementedError

    def extract_disp(self, idata):
        """Extract the dispsersion parameter from a Gaussian posterior."""

        sigma = idata.posterior[self.model.response.name + "_sigma"].values
        sigma = sigma[:, :, np.newaxis]
        sigma = torch.from_numpy(sigma).float()
        return sigma
