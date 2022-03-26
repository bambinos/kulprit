"""Utility functions module."""

import dataclasses

import torch

import arviz as az


def _build_restricted_model(full_model, model_size=None):
    """Build a restricted model from a full model given some set of covariates.

    Args:
        full_model (kulprit.ModelData): The full model
        model_size (int): The number of parameters to use in the restricted model

    Returns:
        kulprit.ModelData: A restricted model with only parameters in `var_names`
    """

    if model_size == full_model.num_params:
        # if `model_size` is same as the full model, simply copy the full_model
        return dataclasses.replace(full_model, inferencedata=None, predictions=None)

    # get the variable names of the best model with `model_size` parameters
    print(full_model.var_names, model_size, full_model.var_names[:model_size])
    var_names = full_model.var_names[:model_size]
    keep_vars = [(var in var_names) for var in full_model.var_names]
    X_res = full_model.X[:, keep_vars]
    num_obs, num_params = X_res.shape
    res_model = dataclasses.replace(
        full_model,
        X=X_res,
        num_params=num_params,
        var_names=var_names,
        inferencedata=None,
        predictions=None,
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
    """Convert some set of pytorch tensors into an ArviZ InferenceData object.

    Args:
        theta_perp (torch.tensor): Restricted parameter posterior projections,
            including the intercept term
        model (kulprit.ModelData): The restricted ModelData object whose
            posterior to build
        disp_perp (torch.tensor): Restricted model dispersions parameter
            posterior projections

    Returns:
        arviz.InferenceData: Restricted model posterior
    """

    print(model.var_names, theta_perp.shape)
    var_dict = {
        f"{model.var_names[i]}": theta_perp[:, i] for i in range(len(model.var_names))
    }
    if disp_perp is not None:
        disp_dict = {f"{model.response_name}_sigma": disp_perp}
        var_dict.update(disp_dict)
    idata = az.convert_to_inference_data(var_dict)
    return idata


def _compute_elpd(model):
    """Compute the ELPD LOO estimates from a fitted model.

    Args:
        model (kulprit.ModelData): The model to diagnose

    Returns:
        arviz.ELPDData: The ELPD LOO estimates
    """

    raise NotImplementedError
