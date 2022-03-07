"""Kullback-Leibler divergence module."""

import torch


class KLDiv:
    """Kullback-Leibler divergence functions switch class.

    In the KLDivSurrogateLoss class, we introduce a private method::

    def _div_fun(family, y_ast, y_perp):
        KLDiv.switch(family, y_ast, y_perp)

    and then compute the loss in a module manner with::

    kl_div = _div_fun(self.family, y_ast, y_perp)


    """

    DEFAULT = "_DEFAULT"
    _func_map = {}

    def __init__(self, case):
        self.case = case

    def __call__(self, f):
        self._func_map[self.case] = f
        return f

    @classmethod
    def _default(cls):
        raise NotImplementedError("Unsupported family.")

    @classmethod
    def switch(cls, case, y_ast, y_perp):
        return cls._func_map.get(case, cls._default)(y_ast, y_perp)


@KLDiv("gaussian")
def _gaussian_kl(y_ast, y_perp):
    """Kullback-Leibler divergence between two Gaussians surrogate function.

    Args:
        mu_ast (torch.tensor): Tensor of learned reference model parameters
        mu_perp (torch.tensor): Tensor of submodel parameters to learn

    Returns:
        torch.tensor: Tensor of shape () containing sample KL divergence
    """

    # compute sufficient statistics
    mu_ast, mu_perp = torch.mean(y_ast), torch.mean(y_perp)
    std_ast, std_perp = torch.std(y_ast), torch.std(y_perp)
    # compute KL divergence using full formula
    div = (
        torch.log(std_perp / std_ast)
        + (std_ast**2 + (mu_ast - mu_perp) ** 2) / (2 * std_perp**2)
        - 1 / 2
    )
    assert div.shape == (), f"Expected data dimensions {()}, received {div.shape}."
    return div


@KLDiv("binomial")
def _binomial_kl(y_ast, y_perp):
    """Kullback-Leibler between two Binomials surrogate function.

    To do:
        * Implement function.
    """
    raise NotImplementedError


@KLDiv("poisson")
def _poisson_kl(y_ast, y_perp):
    """Kullback-Leibler between two Poissons surrogate function.

    To do:
        * Implement function.
    """
    raise NotImplementedError
