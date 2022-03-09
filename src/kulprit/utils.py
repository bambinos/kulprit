"""Utility functions module."""

import dataclasses

import torch

import arviz as az


def _build_restricted_model(full_model, cov_names=None):
    """Build a restricted model from a full model given some set of covariates.

    Args:
        full_model (kulprit.ModelData): The full model
        cov_names (list): The names parameters to use in the restricted model

    Returns:
        kulprit.ModelData: A restricted model with only parameters in `cov_names`
    """

    if cov_names == full_model.cov_names or cov_names is None:
        return dataclasses.replace(full_model, posterior=None, predictions=None)

    keep_vars = [(cov in cov_names) for cov in full_model.cov_names]
    if full_model.has_intercept:
        keep_vars = [True] + keep_vars
    X_res = full_model.X[:, keep_vars]
    n, m = X_res.shape
    res_model = dataclasses.replace(
        full_model, X=X_res, m=m, cov_names=cov_names, posterior=None, predictions=None
    )
    return res_model


def _extract_insample_predictions(model):
    """Extract some model's in-sample predictions.

    Args:
        model (kulprit.ModelData): Some model we wish to get predictions from

    Returns:
        torch.tensor: The in-sample predictions
    """

    y_pred = (
        torch.from_numpy(model.predictions.posterior.y_mean.values)
        .float()
        .reshape(model.s, model.n)
    )
    return y_pred


def _extract_theta_perp(solver, cov_names):
    """Extract restricted parameter projections from PyTorch optimisation.

    Args:
        solver: Trained PyTorch optimisation solver object

    Returns:
        arviz.InferenceData: Projected parameter samples
    """

    theta_perp = list(solver.parameters())[0].data
    datadict = {
        "Intercept": theta_perp[:, 0],
    }
    paramdict = {
        f"{cov_names[i]}_perp": theta_perp[:, i + 1] for i in range(len(cov_names))
    }
    datadict.update(paramdict)
    dataset = az.convert_to_inference_data(datadict)
    return dataset
