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

    y_pred = torch.from_numpy(
        model.predictions.stack(samples=("chain", "draw")).values.T
    ).float()
    return y_pred


def _build_posterior(theta_perp, model, disp_perp=None):
    """Convert some set of pytorch tensors into an arViz InferenceData object.

    Args:
        theta_perp (torch.tensor): Restricted parameter posterior projections
        model (kulprit.ModelData): The model whose posterior to build
        disp_perp (torch.tensor): Restricted model dispersions parameter
            posterior projections

    Returns:
        arviz.InferenceData: Restricted model posterior
    """

    data_dict = {
        "Intercept": theta_perp[:, 0],
    }
    cov_dict = {
        f"{model.cov_names[i]}": theta_perp[:, i + 1]
        for i in range(len(model.cov_names))
    }
    data_dict.update(cov_dict)
    if disp_perp is not None:
        disp_dict = {f"{model.response_name}_sigma": disp_perp}
        data_dict.update(disp_dict)
    data_dict.update(cov_dict)
    dataset = az.convert_to_inference_data(data_dict)
    return dataset


def _compute_elpd(model):
    """Compute the ELPD LOO estimates from a fitted model.

    Args:
        model (kulprit.ModelData): The model to diagnose

    Returns:
        arviz.ELPDData: The ELPD LOO estimates
    """

    raise NotImplementedError
