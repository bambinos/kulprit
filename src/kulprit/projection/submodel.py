"""Submodel optimsation module."""

import torch
import torch.nn as nn

from .divergences import KLDiv


class KLDivSurrogateLoss(nn.Module):
    """Custom Kullback-Leibler divergence surrogate loss module.

    This class computes some KL divergence loss surrogate for observations seen
    from the GLM given the reference model variate's family.

    Attributes:
        family (str): The reference model variate's family
    """

    def __init__(self, family, reduction=torch.sum):
        """Loss module constructor.

        Args:
            family (str): The reference model variate's family
        """
        super().__init__()
        self.family = family
        self.reduction = reduction

    def _div_fun(self, family, y_ast, y_perp):
        """Switch function for KL divergence surrogate."""
        return KLDiv.switch(family, y_ast, y_perp)

    def forward(self, y_ast, y_perp):
        """Forward method in learning loop.

        This method computes the Kullback-Leibler divergence between the
        reference model variate draws ``y_ast``and the restricted model's
        variate draws ``y_perp``. This is done using the two samples' respective
        sufficient sample statistics and a surrogate divergence equation found
        in the ``KLDiv`` class.

        Args:
            mu_ast (torch.tensor): Tensor of learned reference model parameters
            mu_perp (torch.tensor): Tensor of submodel parameters to learn

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence

        Raises:
            AssertionError if unexpected input dimensions
        """

        divs = self._div_fun(self.family, y_ast, y_perp)
        return divs


class SubModel(nn.Module):
    """Core submodel solver class.

    This class solves the general problem of Kullback-Leibler divergence
    projection onto a submodel using a PyTorch neural network architecture
    for efficiency. The procedure might use this class to project the
    learned full parameter samples onto a submodel that uses a restricted
    dataset to define which parameters to include/exclude.

    Attributes:
        inv_link (function): The inverse link function of the GLM
        s (int): Number of MCMC posterior samples
        n (int): Number of observations in the GLM
        m (int): Number of parameters in the submodel
        lin (torch.nn module): The linear transformation module

    Methods:
        forward: performs the forward step of the module

    Todo:
        * Find way to optimise without needing a loss reduction
    """

    def __init__(self, inv_link, s, n, m):
        """SubModel class constructor method.

        Args:
            inv_link (function): The inverse link function of the GLM
            s (int): Number of MCMC posterior samples
            n (int): Number of observations in the GLM
            m (int): Number of parameters in the submodel
        """

        super().__init__()
        # assign data shapes and GLM inverse link function
        self.s = s
        self.n = n
        self.m = m
        self.inv_link = inv_link
        # build linear component of GLM without intercept
        self.lin = nn.Linear(self.m, self.s, bias=False)

    def forward(self, X):
        """Forward method in learning loop.

        Args:
            X (torch.tensor): Design matrix (including intercept) of shape (n, m)

        Returns:
            y (torch.tensor): Model outputs of shape (n, s)

        Raises:
            AssertionError if unexpected input dimensions
        """
        assert X.shape == (
            self.n,
            self.m,
        ), f"Expected data dimensions {(self.n, self.m)}, received {X.shape}."
        # perform forward prediction step
        y = self.inv_link(self.lin.forward(X).T)
        assert y.shape == (
            self.s,
            self.n,
        ), f"Expected variates dimensions {(self.s, self.n)}, received {y.shape}."
        return y
