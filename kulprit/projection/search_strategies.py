"""This module contains the search strategies"""

from kulprit.projection.arviz_io import compute_loo


def user_path(_project, path):
    submodels = []
    for term_names in path:
        submodel = _project(term_names)
        compute_loo(submodel=submodel)
        submodels.append(submodel)
    return submodels


def forward_search(_project, ref_terms, max_terms):
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
        candidates = get_candidates(submodel.term_names, ref_terms)
        projections = [_project(candidate) for candidate in candidates]

        # identify the best candidate by loss (equivalent to KL min)
        submodel = min(projections, key=lambda projection: projection.loss)

        # compute loo for the best candidate and update inplace
        compute_loo(submodel=submodel)

        # add best candidate to the list of selected submodels
        submodels.append(submodel)

    return submodels


def get_candidates(prev_subset, ref_terms):
    """Method for extracting a list of all candidate submodels.

    Parameters:
    ----------
        k : int
    The number of terms in the previous submodel, from which we wish to find all
    possible candidate submodels.

    Returns:
    -------
        List: A list of lists, each containing the terms of all candidate submodels
    """

    candidate_additions = list(set(ref_terms).difference(prev_subset))
    candidates = [prev_subset + [addition] for addition in candidate_additions]
    return candidates
