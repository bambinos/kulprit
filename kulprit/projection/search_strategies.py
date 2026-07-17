"""This module contains the search strategies"""

from itertools import combinations
import numpy as np

try:
    from sklearn.linear_model import lasso_path
except ImportError:
    pass
from kulprit.projection.arviz_io import compute_loo


def user_path(_project, user_terms, refmodel=None):
    submodels = []
    for term_names in user_terms:
        submodel = _project(term_names, clusters=False)
        compute_loo(submodel=submodel, refmodel=refmodel)
        submodels.append(submodel)
    return submodels


def forward_search(_project, ref_terms, max_terms, refmodel, early_stop, requiere_lower_terms):
    """Method for performing forward search.

    Parameters:
    ----------
    _project : function
        A function that takes a list of term names and returns a submodel object.
    ref_terms : list
        A list of all terms in the model.
    max_terms : int
        The maximum number of terms to include in the submodel.
    refmodel : RefModel
        The reference model.
    early_stop : str
        The early stopping criterion. Either "mean" or "se".
    requiere_lower_terms : bool
        Include higher-order interactions only if all lower-order interactions and main effects
        are already in the subset. Defaults to True.

    Returns:
    -------
    List: A list of submodels, each containing the terms of the submodel and its ELPD.
    """

    # initial intercept-only subset
    submodel_size = 0
    term_names = []
    submodel = _project(term_names, clusters=False)
    compute_loo(submodel=submodel, refmodel=refmodel)
    submodels = [submodel]

    while submodel_size < max_terms:
        # increment submodel size
        submodel_size += 1

        # get list of candidate submodels, project onto them, and compute
        # their distances
        candidates = _get_candidates(submodel.term_names, ref_terms, requiere_lower_terms)
        projections = [_project(candidate, clusters=True) for candidate in candidates]

        # identify the best candidate by loss (equivalent to KL min)
        submodel = min(projections, key=lambda projection: projection.loss)

        # reproject the best candidate using more samples
        submodel = _project(submodel.term_names, clusters=False)

        # compute loo for the best candidate and update inplace
        compute_loo(submodel=submodel, refmodel=refmodel)

        if _early_stopping(submodel, refmodel.elpd, early_stop):
            submodels.append(submodel)
            break

        # add best candidate to the list of selected submodels
        submodels.append(submodel)

    return submodels


def l1_search(_project, model, ref_terms, max_terms, refmodel, early_stop):
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
    refmodel : RefModel
        The reference model.
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
        submodel = _project(term_names, clusters=False)

        # compute loo for the best candidate and update inplace
        compute_loo(submodel=submodel, refmodel=refmodel)

        if _early_stopping(submodel, refmodel.elpd, early_stop):
            submodels.append(submodel)
            break

        # add best candidate to the list of selected submodels
        submodels.append(submodel)

        submodel_size += 1
    return submodels


def _get_candidates(prev_subset, ref_terms, requiere_lower_terms=True):
    """Method for extracting a list of all candidate submodels.

    For interaction terms, ensures that all lower-order interactions and main effects
    are included in the previous subset before considering the interaction as a candidate.

    Parameters:
    ----------
    prev_subset : list
        The terms of the previous submodel.
    ref_terms : list
        The terms of the reference model.
    requiere_lower_terms : bool
        Include higher-order interactions only if all lower-order interactions and main effects
        are already in the subset. Defaults to True.

    Returns:
    -------
        List: A list of lists, each containing the terms of all candidate submodels
    """
    prev_subset_set = set(prev_subset)
    candidate_additions = []

    for term in ref_terms:  # pylint: disable=too-many-nested-blocks
        # skip terms already in the previous subset
        if term in prev_subset_set:
            continue

        if requiere_lower_terms:
            # group-specific terms require their fixed-effect term to be present
            if _is_group_specific_term(term):
                missing = _missing_fixed_effect_for_group_term(term, prev_subset)
                if missing:
                    continue

            # interaction checks are done on the fixed-effect part.
            # e.g. for "A:B|group" we check lower-order terms of "A:B".
            interaction_term = _get_fixed_part(term)
            if ":" in interaction_term:
                missing = _missing_lower_order_terms(interaction_term, prev_subset)
                if missing:
                    continue

        candidate_additions.append(term)

    candidates = [prev_subset + [addition] for addition in candidate_additions]
    return candidates


def _missing_fixed_effect_for_group_term(group_term, term_list):
    """Return the missing fixed-effect term required by a group-specific term.

    For example, 'Days|Subject' requires 'Days' to be present. '1|Subject'
    only requires the intercept, which is always assumed present.
    """
    fixed_part = _get_fixed_part(group_term)
    if fixed_part == "1":
        return set()
    if fixed_part not in term_list:
        return {fixed_part}
    return set()


def _is_group_specific_term(term_name):
    """Return True when the term represents a group-specific effect."""
    return "|" in term_name


def _get_fixed_part(term_name):
    """Return the fixed-effect part of a term.

    For group-specific terms this strips the grouping part, e.g.:
    - "Days|Subject" -> "Days"
    - "A:B | group" -> "A:B"
    For non group-specific terms it returns the term unchanged (trimmed).
    """
    if not _is_group_specific_term(term_name):
        return term_name.strip()
    fixed_part, _ = term_name.split("|", maxsplit=1)
    return fixed_part.strip()


def _missing_lower_order_terms(interaction_term, term_list):
    """Return a set of missing lower-order terms for a given interaction term."""
    factors = interaction_term.split(":")
    missing = set()

    for k in range(1, len(factors)):
        for combo in combinations(factors, k):
            lower_term = ":".join(combo)
            if lower_term not in term_list:
                missing.add(lower_term)
    return missing


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

    if isinstance(early_stop, int):
        if submodel.size == early_stop:
            return True
    if early_stop == "mean":
        if elpd_ref - submodel.elpd <= 4:
            return True
    elif early_stop == "se":
        if submodel.elpd + submodel.elpd_se >= elpd_ref:
            return True
    return False
