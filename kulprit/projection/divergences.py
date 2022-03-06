"""Kullback-Leibler divergence module."""

import torch


class KLDiv:
    """Kullback-Leibler divergence functions switch class.

    In the KLDivSurrogateLoss class, we introduce a private method
    ```python
    def _div_fun(family):
        KLDiv.switch(family)
    ```
    and then compute the loss in a module manner with
    ```python
    kl_div = _div_fun(self.family, ...)
    ```

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
    def switch(cls, case, mu_ast, mu_perp):
        return cls._func_map.get(case, cls._default)(mu_ast, mu_perp)


@KLDiv("gaussian")
def _gaussian_kl(mu_ast, mu_perp):
    """Kullback-Leibler divergence between two Gaussians surrogate function.

    To do:
        * Fix this method to return the true KL divergence, currently the
            reduction returns an untrue value.

    Args:
        mu_ast (torch.tensor): Tensor of learned reference model parameters
        mu_perp (torch.tensor): Tensor of submodel parameters to learn

    Returns:
        torch.tensor: (TODO)
    """
    div = torch.sum(mu_perp - mu_ast, dim=-1).reshape(-1) ** 2
    s = mu_ast.shape[0]
    assert div.shape == torch.Size(
        [s]
    ), f"Expected data dimensions {(s)}, received {div.shape}."
    return div


@KLDiv("binomial")
def _binomial_kl():
    """Kullback-Leibler between two Binomials surrogate function.

    To do:
        * Implement function.
    """
    raise NotImplementedError


@KLDiv("poisson")
def _poisson_kl():
    """Kullback-Leibler between two Poissons surrogate function.

    To do:
        * Implement function.
    """
    raise NotImplementedError
