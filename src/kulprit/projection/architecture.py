"""Restricted model projection optimiser module."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..data import ModelData
from ..data.structure import ModelStructure


class Architecture(ABC, nn.Module):
    """Base optimiser class."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):  # pragma: no cover
        pass


class GLMArchitecture(Architecture):
    """Core optimisation solver class.

    This class solves the general problem of Kullback-Leibler divergence
    projection onto a submodel using a PyTorch neural network architecture
    for efficiency. The procedure might use this class to project the
    learned full parameter samples onto a submodel that uses a restricted
    dataset to define which parameters to include/exclude.

    Attributes:
        inv_link (function): The inverse link function of the GLM
        num_obs (int): Number of observations in the GLM
        num_terms (int): Number of parameters in the submodel
        num_draws (int): Number of MCMC posterior samples
        lin (torch.nn module): The linear transformation module
    """

    def __init__(self, structure: ModelStructure) -> None:
        """GLM architecture class for forward propagation in optimisation.

        Args:
            structure (kulprit.data.ModelStructure): The structure object of the
                submodel
        """

        super().__init__()

        # assign data shapes and GLM inverse link function
        self.num_obs = structure.num_obs
        self.num_terms = structure.num_terms
        self.num_draws = structure.num_draws
        self.inv_link = structure.link.linkinv
        # build linear component of GLM without intercept
        self.lin = nn.Linear(structure.num_terms, structure.num_draws, bias=False)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward method in learning loop.

        Args:
            X (torch.tensor): Design matrix (including intercept) of shape
                (num_obs, num_terms)

        Returns:
            y (torch.tensor): Model outputs of shape (num_obs, num_draws)
        """

        # perform forward prediction step
        y = self.inv_link(self.lin.forward(X).T)
        return y
