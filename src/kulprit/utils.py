"""Utility functions module."""

import dataclasses

import arviz as az


def _build_restricted_model(full_model, cov_names):
    """Build a restricted model from a full model given some set of covariates.

    Args:
        full_model (kulprit._Prit): The full model
        cov_names (list): The names parameters to use in the restricted model

    Returns:
        kulprit._Prit: A restricted model with only parameters in `cov_names`
    """

    if cov_names == full_model.cov_names:
        return dataclasses.replace(full_model, posterior=None, predictions=None)

    res_model = dataclasses.replace(full_model, posterior=None, predictions=None)
    return res_model


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
