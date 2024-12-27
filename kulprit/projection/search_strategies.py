"""This module contains the search strategies"""

import numpy as np
from sklearn.linear_model import lasso_path
from kulprit.projection.arviz_io import compute_loo


def user_path(_project, path):
    submodels = []
    for term_names in path:
        submodel = _project(term_names)
        compute_loo(submodel=submodel)
        submodels.append(submodel)
    return submodels


def forward_search(_project, ref_terms, max_terms, elpd_ref, early_stop):
    """Method for performing forward search.

    Parameters:
    ----------
    _project : function
        A function that takes a list of term names and returns a submodel object.
    ref_terms : list
        A list of all terms in the model.
    max_terms : int
        The maximum number of terms to include in the submodel.
    elpd_ref : float
        The expected log pointwise predictive density of the reference model.
    early_stop : str
        The early stopping criterion. Either "mean" or "se".

    Returns:
    -------
    List: A list of submodels, each containing the terms of the submodel and its ELPD.
    """

    # initial intercept-only subset
    submodel_size = 0
    term_names = []
    submodel = _project(term_names)
    compute_loo(submodel=submodel)
    submodels = [submodel]

    while submodel_size < max_terms:
        # increment submodel size
        submodel_size += 1

        # get list of candidate submodels, project onto them, and compute
        # their distances
        candidates = _get_candidates(submodel.term_names, ref_terms)
        projections = [_project(candidate) for candidate in candidates]

        # identify the best candidate by loss (equivalent to KL min)
        submodel = min(projections, key=lambda projection: projection.loss)

        # compute loo for the best candidate and update inplace
        compute_loo(submodel=submodel)

        if _early_stopping(submodel, elpd_ref, early_stop):
            submodels.append(submodel)
            break

        # add best candidate to the list of selected submodels
        submodels.append(submodel)

    return submodels


def l1_search(_project, model, ref_terms, max_terms, elpd_ref, early_stop):
    """Method for performing l1 search.

    Parameters:
    ----------
    _project : function
        A function that takes a list of term names and returns a submodel object.
    model : object
        A Bambi model.
    ref_terms : list
        A list of all terms in the model.
    max_terms : int
        The maximum number of terms to include in the submodel.
    elpd_ref : float
        The expected log pointwise predictive density of the reference model.
    early_stop : str
        The early stopping criterion. Either "mean" or "se".

    Returns:
    -------
    List: A list of submodels, each containing the terms of the submodel and its ELPD.
    """

    d_component = model.distributional_components[model.family.likelihood.parent]
    X = np.column_stack([d_component.design.common[term] for term in ref_terms])
    # XXX we need to make this more general  # pylint: disable=fixme
    mean_param_name = list(model.family.link.keys())[0]
    eta = model.family.link[mean_param_name].link(
        model.components[model.family.likelihood.parent].design.response.design_matrix
    )
    # compute L1 path in the latent space
    _, coef_path, *_ = lasso_path(X, eta)
    cov_order = _first_non_zero_idx(coef_path)

    # sort the covariates according to their L1 ordering
    cov_lasso = dict(sorted(cov_order.items(), key=lambda item: item[1]))
    sorted_covs = [ref_terms[k] for k in cov_lasso]

    submodel_size = 0
    submodels = []

    while submodel_size < max_terms + 1:
        term_names = sorted_covs[:submodel_size]
        submodel = _project(term_names)

        # compute loo for the best candidate and update inplace
        compute_loo(submodel=submodel)

        if _early_stopping(submodel, elpd_ref, early_stop):
            submodels.append(submodel)
            break

        # add best candidate to the list of selected submodels
        submodels.append(submodel)

        submodel_size += 1
    return submodels


def _get_candidates(prev_subset, ref_terms):
    """Method for extracting a list of all candidate submodels.

    Parameters:
    ----------
    prev_subset : list
        The terms of the previous submodel.
    ref_terms : list
        The terms of the reference model.

    Returns:
    -------
        List: A list of lists, each containing the terms of all candidate submodels
    """

    candidate_additions = list(set(ref_terms).difference(prev_subset))
    candidates = [prev_subset + [addition] for addition in candidate_additions]
    return candidates


def _first_non_zero_idx(arr):
    """Find the index of the first non-zero element in each row of a matrix.

    Parameters:
    ----------
    arr : np.ndarray
        A matrix.

    Returns:
    -------
    dict: Dictionary keyed by the row number where each value is the index of the first
    non-zero element in that row.
    """

    # initialise dictionary of indices
    idx_dict = {}

    # loop through each row and find first non-zero element
    for i, j in zip(*np.where(arr != 0)):
        if i in idx_dict:
            continue
        idx_dict[i] = j

    # identify which keys are missing and set their values to infinity
    if len(idx_dict) < arr.shape[0]:
        missing_keys = set(range(arr.shape[0])) - set(idx_dict.keys())
        for key in missing_keys:
            idx_dict[key] = np.inf

    return idx_dict


def _early_stopping(submodel, elpd_ref, early_stop):
    if early_stop == "mean":
        if elpd_ref.elpd_loo - submodel.elpd_loo <= 4:
            return True
    elif early_stop == "se":
        if submodel.elpd_loo + submodel.elpd_se >= elpd_ref.elpd_loo:
            return True
    return False
