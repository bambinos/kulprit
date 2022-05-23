"""Kullback-Leibler divergence losses module."""

import torch

from kulprit.families.family import Family
from kulprit.projection.losses import Loss


class KullbackLeiblerLoss:
    """Kullback-Leibler (KL) divergence loss module.

    This class computes some KL divergence loss for observations seen from the
    the reference model variate's family. The KL divergence is the originally
    motivated loss function by Goutis and Robert (1998).
    """

    def __init__(self, family: Family) -> None:
        """Loss module constructor.

        Args:
            family (Family): Reference model family object
        """

        # log family and retrieve family name
        self.family = family
        self.family_name = family.family.name

        # define all available KL divergence loss classes
        self.loss_dict = {
            "gaussian": GaussianKullbackLeiblerLoss,
        }

        if self.family_name not in self.loss_dict:
            raise NotImplementedError(
                f"The {self.family_name} class has not yet been implemented."
            )

    def factory_method(self) -> Loss:
        """Choose the appropriate divergence class given the model."""

        # return appropriate divergence class given model variate family
        loss_class = self.loss_dict[self.family_name]
        return loss_class()

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

        loss_class = self.factory_method()
        loss = loss_class.forward(P, Q)
        return loss


class GaussianKullbackLeiblerLoss(Loss):
    """Gaussian empirical KL divergence class."""

    def forward(self, P: torch.tensor, Q: torch.tensor) -> torch.tensor:
        """Kullback-Leibler divergence between two Gaussians.

        Args:
            P (torch.tensor): Tensor of reference model posterior parameter
                draws
            Q (torch.tensor): Tensor of submodel posterior parameter draws

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """

        # compute KL divergence loss
        loss = torch.mean(torch.abs(P - Q) ** 2) ** (1 / 2)

        assert loss.shape == (), f"Expected data dimensions {()}, received {loss.shape}."
        return loss
