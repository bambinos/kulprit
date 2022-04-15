"""Utility functions module."""

import arviz as az
from arviz.utils import one_de
import numpy as np
import torch


def _compute_log_likelihood(backend, points):
    """Compute log-likelihood of some data points given a PyMC model.

    Args:
        backend (pymc.Model) : PyMC3 model for which to compute log-likelihood
        points (list) : List of dictionaries, where each dictionary is a named
            sample of all parameters in the model

    Returns:
        dict: Dictionary of log-liklelihoods at each point
    """

    cached = [(var, var.logp_elemwise) for var in backend.model.observed_RVs]

    log_likelihood_dict = {}
    for var, log_like_fun in cached:
        log_like = np.array([one_de(log_like_fun(point)) for point in points])
        log_likelihood_dict[var.name] = log_like
    return log_likelihood_dict


def _extract_insample_predictions(model):
    """Extract some model's in-sample predictions.

    To do:
        * consider moving this method either into the base class, or into a
            separate utility function collection for sparsity

    Args:
        model (kulprit.ModelData): Some model we wish to get predictions from

    Returns:
        torch.tensor: The in-sample predictions
    """

    y_pred = torch.from_numpy(
        model.predictions.stack(samples=("chain", "draw")).values.T
    ).float()
    return y_pred
