import copy
from typing import List

from arviz import InferenceData
from bambi import Model
import numpy as np
import xarray as xr

from kulprit.families.family import Family


def init_idata(
    ref_model: Model,
    ref_idata: InferenceData,
    term_names: List[str] = None,
    num_thinned_samples: int = 400,
):
    """Initialise a submodel InferenceData object including only certain terms.

    Args:
        ref_model (bambi.models.Model): The reference Bambi model to inherit
            from
        ref_idata (arviz.InferenceData): The fitted reference Bambi model's
            ``InferenceData`` object to modify and inherit from
        term_names (list): The names of the terms to include in the submodel
        num_thinned_samples (int): The number of samples to use in thinned
            optimisation

    Returns:
        arviz.InferenceData: An InferenceData object for the submodel inheriting
            from the fitted reference model and including only information
            pertaining to certain terms

    To do:
        * Allow this method to take a ``ref_model`` different to the one which
            produced ``ref_idata``. In a word, allow the procedure to operate
            on submodel parameter spaces that are disjoint to the reference
            model parameter space
    """

    # copy term names so as not to modify input variables
    term_names = copy.deepcopy(term_names)

    # set default projection
    if not term_names:
        term_names = ref_model.term_names

    # initialise family object
    family = Family(model=ref_model)
    if family.has_dispersion_parameters:
        term_names.append(family.disp_name)

    # make copy of the reference model's inference data
    res_idata = copy.deepcopy(ref_idata)
    del res_idata.log_likelihood, res_idata.sample_stats

    # extract dimensions from reference model's inference data
    num_chain = ref_idata.posterior.dims["chain"]
    num_draw = ref_idata.posterior.dims["draw"]
    num_samples = num_chain * num_draw

    # produce thinned indices
    thinned_idx = np.random.randint(0, num_samples, num_thinned_samples)

    # extract thinned parameters
    new_data_vars = {
        name: (list(var.dims), var.values[thinned_idx])
        for name, var in ref_idata.posterior.transpose(*("chain", "draw", ...))
        .stack(samples=("chain", "draw"))[term_names]
        .transpose(*("samples", ...))
        .data_vars.items()
    }

    # define new posterior dimensions
    new_dims = dict(ref_idata.posterior.dims)
    new_dims["draw"] = 100
    new_coords = {
        name: np.arange(stop=new_dims[name], step=1) for name in new_dims.keys()
    }

    # build attributes dictionary
    res_attrs = {
        "term_names": list(ref_model.term_names),
        "common_terms": list(ref_model.common_terms.keys()),
        "response_name": ref_model.response.name,
        "has_intercept": ref_model.intercept_term is not None,
        "model_size": len(ref_model.common_terms),
        "num_obs": ref_model._design.common.design_matrix.shape[0],
        "num_terms": ref_model._design.common.design_matrix.shape[1],
        "family": ref_model.family,
    }

    # build restricted posterior object and replace old one
    res_posterior = xr.Dataset(
        data_vars=new_data_vars, coords=new_coords, attrs=res_attrs
    )
    res_idata.posterior = res_posterior.transpose(*("draw", ...))

    # unstack dimensions
    res_idata.stack(samples=["chain", "draw"], inplace=True)
    res_idata.unstack(inplace=True)
    return res_idata
