"""Model architecture module."""

import torch
import torch.nn as nn

from kulprit.data.submodel import SubModelStructure


class BaseArchitecture(nn.Module):
    """Base optimiser class."""

    def __init__(self):
        super(BaseArchitecture, self).__init__()


class Architecture:
    """Architecture factory class."""

    def __init__(self, submodel_structure: SubModelStructure) -> None:
        """Architecture factory constructor.

        Args:
            submodel_structure (SubModelStructure): The submodel stucture data
                object
        """

        # log model data
        self.submodel_structure = submodel_structure

        # define all available KL divergence loss classes
        self.architecture_dict = {
            "glm": GLMArchitecture,
        }

        if self.submodel_structure.architecture not in self.architecture_dict:
            raise NotImplementedError(
                f"The {self.submodel_structure.architecture} architecture has "
                + "not yet been implemented."
            )

        # build architecture class
        self.architecture = self.factory_method()

    def factory_method(self) -> BaseArchitecture:
        """Choose the appropriate architecture class given the model."""

        # return appropriate divergence class given model variate family
        architecture_class = self.architecture_dict[self.submodel_structure.architecture]
        return architecture_class(self.submodel_structure)


class GLMArchitecture(BaseArchitecture):
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

    def __init__(self, submodel_structure: SubModelStructure) -> None:
        """GLM architecture class for forward propagation in optimisation.

        Args:
            submodel_structure (kulprit.data.SubModelStructure): The structure object of
                the submodel
        """

        super(GLMArchitecture, self).__init__()

        # assign data shapes and GLM inverse link function
        self.num_obs = submodel_structure.num_obs
        self.num_terms = submodel_structure.num_terms
        self.num_draws = submodel_structure.num_draws
        self.inv_link = submodel_structure.link.linkinv
        # build linear component of GLM without intercept
        self.lin = nn.Linear(
            submodel_structure.num_terms, submodel_structure.num_draws, bias=False
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward method in learning loop.

        Args:
            X (torch.tensor): Design matrix (including intercept) of shape
                (num_obs, num_terms)

        Returns:
            torch.tensor: Model outputs of shape (num_obs, num_draws)
        """

        # perform forward prediction step
        out = self.inv_link(self.lin.forward(X).T)
        return out
