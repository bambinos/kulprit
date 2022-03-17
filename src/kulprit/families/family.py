"""Kullback-Leibler divergence module."""

import abc

import torch

from ..utils import _extract_theta_perp, _extract_theta_ast, _extract_sigma_ast


class Family(abc.ABC):
    """Kullback-Leibler divergence functions switch class."""

    subclasses = {}

    def __init__(self):
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._FAMILY_NAME] = cls

    @abc.abstractmethod
    def kl_div(self):  # pragma: no cover
        pass

    @classmethod
    def create(cls, model):
        if model.family.name not in cls.subclasses:
            raise NotImplementedError("Unsupported family.")

        return cls.subclasses[model.family.name](model)


class Gaussian(Family):
    _FAMILY_NAME = "gaussian"

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.has_disp_params = True

    def kl_div(self, y_ast, y_perp):
        """Kullback-Leibler divergence between two Gaussians.

        Args:
            y_ast (torch.tensor): Tensor of reference model posterior draws
            y_perp (torch.tensor): Tensor of restricted model posterior draws

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """

        # compute Wasserstein distance as a KL divergence surrogate
        div = torch.mean((y_ast - y_perp) ** 2)
        assert div.shape == (), f"Expected data dimensions {()}, received {div.shape}."
        return div

    def _project_disp_params(self, res_model):
        """Analytic projection of the model dispersion parameters.

        Args:
            res_model (kulprit.ModelData): Restricted model on which to project
                the reference dispersion parameters

        Returns:
            torch.tensor: The restricted projections of the dispersion parameters
        """

        def _proj(theta_ast, theta_perp):
            """Projection method to aid with vectorisation."""

            f = X_ast @ theta_ast
            f_perp = X_perp @ theta_perp
            sigma_perp = torch.sqrt(
                sigma_ast**2 + 1 / self.ref_model.n * (f - f_perp).T @ (f - f_perp)
            )
            return sigma_perp

        # extract design matrix and parameter draws from both models
        theta_ast = _extract_theta_ast(self.ref_model)
        sigma_ast = _extract_sigma_ast(self.ref_model)
        theta_perp = _extract_theta_perp(res_model)
        X_ast = self.ref_model.X
        X_perp = res_model.X
        # vectorise the projection method and perform sample-wise projection
        _proj_vmap = torch.vmap(_proj)
        sigma_perp = _proj_vmap(theta_ast, theta_perp)
        return sigma_perp
