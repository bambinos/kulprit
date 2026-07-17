"""Functions to interact with PyMC models"""

import warnings
import numpy as np

from pymc import do, compute_log_likelihood
from pymc.logprob.utils import ParameterValueError
from pymc.util import is_transformed_name, get_untransformed_name
from pymc.pytensorf import join_nonshared_inputs
from pytensor import function, shared
from pytensor.tensor import tensor
from pytensor.graph import ancestors


def compile_mllk(model, initial_point):
    """
    Compile the log-likelihood function for the model to be able to condition on both
    data and parameters.
    """
    obs_rvs = model.observed_RVs[0]
    new_y_value = obs_rvs.type()
    original_value = model.rvs_to_values.get(obs_rvs)
    model.rvs_to_values[obs_rvs] = new_y_value

    vars_ = model.value_vars

    try:
        [logp], raveled_inp = join_nonshared_inputs(
            point=initial_point, outputs=[model.datalogp], inputs=vars_
        )
        rv_logp_fn = function([raveled_inp, new_y_value], logp)
        rv_logp_fn.trust_input = True

        def fmodel(params, *pred):
            try:
                if len(pred) == 2:
                    return -(rv_logp_fn(params, pred[0]) + rv_logp_fn(params, pred[1]))
                return -(rv_logp_fn(params, pred[0]))
            except ParameterValueError:
                # Handle floating-point edge cases underflow/overflow for extreme values on the
                #  transformed unbounded scale.
                return np.inf

        return fmodel
    finally:
        if original_value is None:
            del model.rvs_to_values[obs_rvs]
        else:
            model.rvs_to_values[obs_rvs] = original_value


def turn_off_terms(switches, all_terms, term_names):
    """
    Turn off the terms not in term_names
    """
    for term in all_terms:
        if term not in term_names:
            switches[term].set_value(0.0)
        else:
            switches[term].set_value(1.0)


def add_switches(model, ref_terms):
    switches = {term: shared(1.0) for term in ref_terms}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Intervention expression references")
        switched_terms = {term: model.named_vars[term] * switches[term] for term in ref_terms}
        return do(model, switched_terms), switches


def compute_llk(idata, model):
    """Compute log-likelihood for the submodel."""
    return compute_log_likelihood(idata, model=model, progressbar=False, extend_inferencedata=False)


def get_term_variables(model, term_names):
    """
    Map each switchable term name to the set of free-RV names that belong to it.

    This works for both common terms (the coefficient is a free RV) and
    group-specific terms (non-centered: offset + sigma; centered: coefficient + sigma).

    For group-specific terms the random-effect standard deviation ``sigma`` is
    excluded. The scale of the random effects is not identified by the likelihood
    alone, so optimizing sigma jointly with the coefficients leads to numerical
    instability. It is kept frozen at its initial value.
    """
    free_rv_names = {rv.name for rv in model.free_RVs}
    term_to_vars = {}
    for term_name in term_names:
        if term_name not in model.named_vars:
            raise KeyError(
                f"Term '{term_name}' is not available in model.named_vars. "
                "This can happen when term aliases do not match backend variable names."
            )
        term_var = model.named_vars[term_name]
        anc_names = {var.name for var in ancestors([term_var])}
        # include the term variable itself when it is a free RV (centered case)
        if term_var.name in free_rv_names:
            anc_names.add(term_var.name)
        vars_ = anc_names & free_rv_names
        # freeze the random-effect standard deviation at its initial value.
        # This also excludes any further free RVs that feed into sigma itself
        # (e.g. a hierarchical/nested prior on sigma).
        if "|" in term_name:
            sigma_name = f"{term_name}_sigma"
            excluded = {sigma_name}
            if sigma_name in model.named_vars:
                sigma_var = model.named_vars[sigma_name]
                excluded |= {var.name for var in ancestors([sigma_var])}
            vars_ = {v for v in vars_ if v not in excluded}
        if not vars_:
            raise ValueError(
                f"Term '{term_name}' does not map to any optimizable free RV after filtering."
            )
        term_to_vars[term_name] = vars_
    return term_to_vars


def get_model_information(model, initial_point):
    """
    Get the size and transformation of each variable in a PyMC model.
    """

    var_info = {}
    for v_var in model.value_vars:
        name = v_var.name
        if is_transformed_name(name):
            name = get_untransformed_name(name)
            ndim = initial_point[v_var.name].ndim
            # PyTensor may interpret names starting with digits or containing '|' as dtype
            # strings, so sanitize the variable name.
            safe_name = name.replace("|", "_").replace(":", "_").replace(" ", "_")
            x_var = tensor(
                dtype="float64", shape=(None,) * (ndim + 2), name=f"transformed_{safe_name}"
            )
            z_var = model.rvs_to_transforms[model.values_to_rvs[v_var]].backward(x_var)
            transformation = function(inputs=[x_var], outputs=z_var)
        else:
            transformation = None

        var_info[name] = (
            initial_point[v_var.name].shape,
            initial_point[v_var.name].size,
            transformation,
        )

    return var_info
