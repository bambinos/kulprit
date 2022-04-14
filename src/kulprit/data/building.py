"""Dataclass building utilities module."""

import dataclasses

import torch
import numpy as np

import arviz as az


def _build_restricted_model(ref_model, model_size=None):
    """Build a restricted model from a reference model given some model size

    Args:
        ref_model (kulprit.ModelData): The reference model
        model_size (int): The number of parameters to use in the restricted model

    Returns:
        kulprit.ModelData: A restricted model with `model_size` terms
    """

    if model_size == ref_model.model_size or model_size is None:
        # if `model_size` is same as the full model, simply copy the ref_model
        return dataclasses.replace(ref_model, inferencedata=None, predictions=None)

    # test model_size in case of misuse
    if model_size < 0 or model_size > ref_model.model_size:
        raise UserWarning(
            "`model_size` parameter must be non-negative and less than size of"
            + f" the reference model, instead received {model_size}."
        )

    # get the variable names of the best model with `model_size` parameters
    restricted_common_terms = ref_model.common_terms[:model_size]
    if model_size > 0:  # pragma: no cover
        # extract the submatrix from the reference model's design matrix
        X_res = torch.from_numpy(
            np.column_stack(
                [ref_model.design.common[term] for term in restricted_common_terms]
            )
        ).float()
        # manually add intercept to new design matrix
        X_res = torch.hstack((torch.ones(ref_model.num_obs, 1), X_res))
    else:
        # intercept-only model
        X_res = torch.ones(ref_model.num_obs, 1).float()

    # update common term names and dimensions and build new ModelData object
    _, num_terms = X_res.shape
    restricted_term_names = ["Intercept"] + restricted_common_terms
    res_model = dataclasses.replace(
        ref_model,
        X=X_res,
        num_terms=num_terms,
        model_size=model_size,
        term_names=restricted_term_names,
        common_terms=restricted_common_terms,
        inferencedata=None,
        predictions=None,
    )
    # ensure correct dimensions
    assert res_model.X.shape == (ref_model.num_obs, model_size + 1)
    return res_model


def _build_idata(theta_perp, model, disp_perp=None):
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

    var_dict = {f"{term}": theta_perp[:, i] for i, term in enumerate(model.term_names)}
    if disp_perp is not None:
        disp_dict = {f"{model.response_name}_sigma": disp_perp}
        var_dict.update(disp_dict)
    idata = az.convert_to_inference_data(var_dict)
    return idata
