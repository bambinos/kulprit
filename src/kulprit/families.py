"""Kullback-Leibler divergence module."""

import abc

import torch


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
    def create(cls, family_name):
        if family_name not in cls.subclasses:
            raise NotImplementedError("Unsupported family.")

        return cls.subclasses[family_name]()


class Gaussian(Family):
    _FAMILY_NAME = "gaussian"

    def __init__(self):
        super().__init__()
        self.has_disp_params = True

    def kl_div(self, y_ast, y_perp):
        """Kullback-Leibler divergence between two Gaussians surrogate function.

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

    def project_disp_params(self):  # pragma: no cover
        "Project the dispersion parameters of the distribution analytically."
