"""Losses module."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..data import ModelData


class Loss(nn.Module, ABC):
    """Base loss class."""

    @abstractmethod
    def forward(self):
        pass


class KullbackLeiblerLoss(Loss):
    """Kullback-Leibler (KL) divergence loss module.

    This class computes some KL divergence loss for observations seen from the
    the reference model variate's family. The KL divergence is the originally
    motivated loss function by Goutis and Robert (1998).
    """

    def __init__(self, data: ModelData) -> None:
        """Loss module constructor.

        Args:
            data (kulprit.data.ModelData): Reference model dataclass object
        """

        super().__init__()

        # define all available KL divergence loss classes
        self.family_dict = {
            "gaussian": GaussianKullbackLeiblerLoss,
        }

        # log family name
        self.family = data.structure.family

        if self.family not in self.family_dict:
            raise NotImplementedError(
                f"The {self.family} class has not yet been implemented."
            )

    def factory_method(self) -> Loss:
        """Choose the appropriate divergence class given the model."""

        # return appropriate divergence class given model variate family
        div_class = self.family_dict[self.family]
        return div_class()

    def forward(self, P: torch.tensor, Q: torch.tensor) -> torch.tensor:
        """Forward method in learning loop.

        This method computes the Kullback-Leibler divergence between the
        reference model variate draws ``P``and the restricted model's variate
        draws ``Q``. This is done using the two samples' respective sufficient
        sample statistics and a divergence equation found in the ``Family``
        class.

        Args:
            P (torch.tensor): Tensor of the reference model posterior MCMC
                draws
            Q (torch.tensor): Tensor of the restricted model posterior MCMC
                draws

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """

        div_class = self.factory_method()
        divs = div_class.forward(P, Q)
        return divs


class GaussianKullbackLeiblerLoss(Loss):
    """Gaussian empirical KL divergence class."""

    def forward(self, P: torch.tensor, Q: torch.tensor) -> torch.tensor:
        """Kullback-Leibler divergence between two Gaussians.

        Args:
            P (torch.tensor): Tensor of reference model posterior draws
            Q (torch.tensor): Tensor of restricted model posterior draws

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """

        # compute Wasserstein distance as a KL divergence surrogate
        div = torch.mean((P - Q) ** 2)

        assert div.shape == (), f"Expected data dimensions {()}, received {div.shape}."
        return div
