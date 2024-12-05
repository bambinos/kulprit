"""Functions to interact with PyMC models"""
import numpy as np

from pymc import do, compute_log_likelihood
from pymc.util import is_transformed_name, get_untransformed_name
from pymc.pytensorf import compile_pymc, join_nonshared_inputs
from pytensor import function
from pytensor.tensor import matrix


def compile_mllk(model):
    """
    Compile the log-likelihood function for the model to be able to condition on both
    data and parameters.
    """
    obs_rvs = model.observed_RVs[0]
    old_y_value = model.rvs_to_values[obs_rvs]  # do we need to restore the value
    new_y_value = obs_rvs.type()
    model.rvs_to_values[obs_rvs] = new_y_value

    vars_ = model.value_vars
    initial_point = model.initial_point()

    [logp], raveled_inp = join_nonshared_inputs(
        point=initial_point, outputs=[model.datalogp], inputs=vars_
    )

    rv_logp_fn = compile_pymc([raveled_inp, new_y_value], logp)
    rv_logp_fn.trust_input = True

    def fmodel(params, obs):
        return -rv_logp_fn(params, obs).sum()

    return fmodel, old_y_value, obs_rvs


def compute_new_model(model, noncentered, ref_var_info, all_terms, term_names):
    """
    Compute a new model by excluding the terms not in term_names.
    """
    print("term_names", term_names)
    # get all the terms not in term_names
    exclude_terms = {term: 0 for term in set(all_terms) - set(term_names)}
    for term in exclude_terms.keys():
        shape = ref_var_info[term][0]
        exclude_terms[term] = np.zeros(shape)

    if noncentered:
        for term in term_names:
            if term.startswith("1|"):
                exclude_terms.pop(term + "_sigma")
            if "|" in term:
                exclude_terms.pop(term + "_offset")

    return do(
        model,
        exclude_terms,
        prune_vars=True,
    )


def compute_llk(idata, model):
    """Compute log-likelihood for the submodel."""
    return compute_log_likelihood(idata, model=model, progressbar=False, extend_inferencedata=False)


def get_model_information(model):
    """
    Get the size and transformation of each variable in a PyMC model.
    """

    var_info = {}
    initial_point = model.initial_point()
    for v_var in model.value_vars:
        name = v_var.name
        if is_transformed_name(name):
            name = get_untransformed_name(name)
            x_var = matrix(f"{name}_transformed")
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
