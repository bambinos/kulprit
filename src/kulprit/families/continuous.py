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

    def kl_div(self, linear_predictor, disp, linear_predictor_ref, disp_ref):
        """Kullback-Leibler divergence between two Gaussians."""

        mean_perp = self.link.linkinv(linear_predictor).mean()
        sigma_perp = disp.mean()
        mean_ref = linear_predictor_ref.mean()
        sigma_ref = disp_ref.mean()

        # compute KL divergence
        loss = (
            torch.log(sigma_ref / sigma_perp)
            + (sigma_perp**2 + (mean_perp - mean_ref) ** 2) / (2 * sigma_ref**2)
            - 0.5
        )
        return loss

    def extract_disp(self, idata):
        """Extract the dispsersion parameter from a Gaussian posterior."""

        sigma = idata.posterior[self.model.response.name + "_sigma"].values
        sigma = sigma[:, :, np.newaxis]
        sigma = torch.from_numpy(sigma).float()
        return sigma
