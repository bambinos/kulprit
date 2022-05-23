"""Core family class to be accessed"""

import torch

from kulprit.data.data import ModelData
from kulprit.data.submodel import SubModelStructure
from kulprit.families import BaseFamily
from kulprit.families.continuous import GaussianFamily


class Family:
    def __init__(self, data: ModelData) -> None:
        """Family factory constructor.

        Args:
            data (kulprit.data.ModelData): Reference model dataclass object
        """

        # log model data and family name
        self.data = data
        self.family_name = data.structure.family

        # define all available family classes
        self.family_dict = {
            "gaussian": GaussianFamily,
        }

        # test family name
        if self.family_name not in self.family_dict:
            raise NotImplementedError(
                f"The {self.family} family has not yet been implemented."
            )

        # build BaseFamily object
        self.family = self.factory_method()

    def factory_method(self) -> BaseFamily:
        """Choose the appropriate family class given the model."""

        # return appropriate family class given model variate family
        family_class = self.family_dict[self.family_name]
        return family_class(self.data)

    def solve_analytic(self, submodel_structure: SubModelStructure) -> torch.tensor:
        """Analytic solution to the reference model parameter projection.

        Args:
            submodel_structure (SubModelStructure): The submodel structure object

        Returns
            tuple: A tuple of the projection solution along with the final loss
                value of the gradient descent
        """

        # test whether or not the family has an analytic solution
        if not self.family.has_analytic_solution:
            return None

        # compute the solution and return
        solution = self.family.solve_analytic(submodel_structure=submodel_structure)
        return solution

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

        # test whether or not the family has dispersion parameters
        if not self.family.has_dispersion_parameters:
            return None

        # compute the solution and return
        solution = self.family.solve_dispersion(theta_perp=theta_perp, X_perp=X_perp)
        return solution
