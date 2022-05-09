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
        log_likelihood = np.array([one_de(log_like_fun(point)) for point in points])
        log_likelihood_dict[var.name] = log_likelihood
    return log_likelihood_dict


def _extract_insample_predictions(model):
    """Extract some model's in-sample predictions.

    Args:
        model (kulprit.ModelData): Some model we wish to get predictions from

    Returns:
        torch.tensor: The in-sample predictions
    """

    dim_order = ("samples", "y_dim_0")
    y_pred = torch.from_numpy(
        model.predictions.stack(samples=("chain", "draw")).transpose(*dim_order).values
    ).float()
    return y_pred
