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
            mu_ast (torch.tensor): Tensor of learned reference model parameters
            mu_perp (torch.tensor): Tensor of submodel parameters to learn

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """

        # compute sufficient statistics
        mu_ast, mu_perp = torch.mean(y_ast, 1), torch.mean(y_perp, 1)
        std_ast, std_perp = torch.std(y_ast, 1), torch.std(y_perp, 1)
        # compute KL divergence using full formula
        div = torch.mean(
            torch.log(std_perp / std_ast)
            + (std_ast**2 + (mu_ast - mu_perp) ** 2) / (2 * std_perp**2)
            - 0.5
        )
        #div = torch.mean((y_ast - y_perp)**2)
        assert div.shape == (), f"Expected data dimensions {()}, received {div.shape}."
        return div

    def project_disp_params(self):  # pragma: no cover
        "Project the dispersion parameters of the distribution analytically."
