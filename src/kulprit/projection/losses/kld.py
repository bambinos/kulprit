"""Kullback-Leibler divergence loss module."""

import torch
import torch.nn.functional as F
from kulprit.families.family import Family

from kulprit.projection.losses import Loss


class KullbackLeiblerLoss(Loss):
    """Kullback-Leibler (KL) divergence loss module.

    This class computes some KL divergence loss for observations seen from the
    the reference model variate's family. The KL divergence is the originally
    motivated loss function by Goutis and Robert (1998).
    """

    def __init__(self, family: Family) -> None:
        """Loss module constructor."""
        super(KullbackLeiblerLoss, self).__init__()

        self.family = family
        self.link = self.family.link

    def forward(
        self, linear_predictor, disp, linear_predictor_ref, disp_ref
    ) -> torch.tensor:
        """Forward method in learning loop.

        This method computes the sample Kullback-Leibler divergence between the
        reference model posterior predictive log probabilities, and those of the
        submodel we wish to optimise. We initially input the raw draws, before
        converting them into log probabilities.

        Args:
            input (torch.tensor): Tensor of the submodel posterior predictive
                draws
            target (torch.tensor): Tensor of the reference model posterior
                predictive draws

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """

        loss = self.family.kl_div(linear_predictor, disp, linear_predictor_ref, disp_ref)
        return loss


def general_kl(input: torch.tensor, target: torch.tensor) -> torch.tensor:
    """General sample-wise Kullback-Leibler loss.

    This method computes the sample Kullback-Leibler divergence between the
    reference model posterior predictive log probabilities, and those of the
    submodel we wish to optimise. We initially input the raw draws, before
    converting them into log probabilities.

    Args:
        input (torch.tensor): Tensor of the submodel posterior predictive
            draws
        target (torch.tensor): Tensor of the reference model posterior
            predictive draws

    Returns:
        torch.tensor: Tensor of shape () containing sample KL divergence
    """

    # transform samples to log probabilities
    input = F.log_softmax(input, dim=-1)
    target = F.log_softmax(target, dim=-1)

    # compute sample-wise KL divergence
    loss_samplewise = target.exp() * (target - input)
    loss_samplewise = loss_samplewise.sum(dim=-1) / input.size(0)

    # reduce over samples
    return loss_samplewise.sum()
