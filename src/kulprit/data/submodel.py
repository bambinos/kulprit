"""Submodel dataclass module."""

from dataclasses import dataclass, field

from typing import Any, List

from arviz import InferenceData
from arviz.data.utils import extract_dataset

from pymc.model import Model
from bambi.backend.pymc import PyMCModel
from pymc.util import is_transformed_name, get_untransformed_name

import xarray as xr

import numpy as np

from kulprit.families.family import Family


@dataclass
class SubModel:
    """Submodel dataclass."""

    idata: InferenceData
    backend: PyMCModel
    kl_div: float
    size: int
    term_names: list
    model: Model = field(init=False)
    model_logp: Any = field(init=False)  # TODO: define datatype
    transforms: Any = field(init=False)  # TODO: define datatype
    num_chain: int = field(init=False)
    num_draw: int = field(init=False)
    num_samples: int = field(init=False)
    num_obs: int = field(init=False)

    def __post_init__(self):
        # extract PyMC model from backend
        self.model = self.backend.model

        # log the compiled model log probability
        self.model_logp = self.model.compile_logp(
            sum=False, vars=self.model.observed_RVs
        )

        # log the transformations
        self.transforms = get_transforms(self.model)

        # extract dimensions
        self.num_chain = self.idata.posterior.dims["chain"]
        self.num_draw = self.idata.posterior.dims["draw"]
        self.num_samples = self.num_chain * self.num_draw
        response_name = list(self.idata.observed_data.data_vars.keys())[0]
        self.num_obs = self.idata.observed_data.dims[f"{response_name}_dim_0"]

    def add_log_likelihood(self):
        # build points data from the posterior dictionaries
        posterior_ = extract_dataset(self.idata).to_dict()
        points = self.posterior_to_points(posterior_)

        # compute log-likelihood of projected model from this posterior
        log_likelihood = self.compute_log_likelihood(self.backend, points)

        # reshape the log-likelihood values to be inline with reference model
        log_likelihood.update(
            (key, value.reshape(self.num_chain, self.num_draw, self.num_obs))
            for key, value in log_likelihood.items()
        )

        # convert dictionary to dataset
        # TODO

        # add to idata object
        self.idata.add_groups({"log_likelihood": log_likelihood})

    def posterior_to_points(self, posterior: dict) -> list:
        """Convert the posterior samples from a restricted model into list of dicts.

        This list of dicts datatype is referred to a `points` in PyMC, and is needed
        to be able to compute the log-likelihood of a projected model, here
        `res_model`.

        Args:
            posterior (dict): Dictionary of posterior restricted model samples

        Returns:
            list: The list of dictionaries of point samples
        """
        initial_point = self.model.initial_point(seed=None)

        points = []
        for i in range(self.num_samples):
            point = {}
            for var, value in initial_point.items():
                if var in posterior.keys():
                    point[var] = posterior[var][i]
                else:
                    point[var] = np.zeros_like(value)
            points.append(point)

        return points

    def compute_log_likelihood(self, backend: Model, points: list) -> dict:
        """Compute log-likelihood of some data points given a PyMC model.

        Args:
            backend (pymc.Model) : PyMC model for which to compute log-likelihood
            points (list) : List of dictionaries, where each dictionary is a named
                sample of all parameters in the model

        Returns:
            dict: Dictionary of log-likelihoods at each point
        """
        log_likelihood_dict = {
            var.name: np.array([self.model_logp(point) for point in points])
            for var in backend.model.observed_RVs
        }

        return log_likelihood_dict


def get_transforms(model):
    """Generate dict with information about transformations

    Args:
        backend (pymc.Model) : PyMC model
    Returns:
        Dictionary with keys unstransformed variable name
        and values (tranformation name, forward transformation)
    """
    transforms = {}
    for var in model.value_vars:
        name = var.name
        transform_name = ""
        transform_function = None
        if is_transformed_name(name):
            name = get_untransformed_name(name)
            transform_name = var.tag.transform.name
            transform_function = var.tag.transform.forward
        transforms[name] = (transform_name, transform_function)
    return transforms


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

    # set default projection
    if not term_names:
        term_names = ref_model.term_names

    # copy term names so as not to modify input variables
    term_names_ = term_names.copy()

    # initialise family object
    family = Family(model=ref_model)
    if family.has_dispersion_parameters:
        term_names_.append(family.disp_name)

    # make copy of the reference model's inference data
    res_idata = ref_idata.copy()
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
        .stack(samples=("chain", "draw"))[term_names_]
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
        "term_names": term_names_,
        "response_name": ref_model.response.name,
        "has_intercept": ref_model.intercept_term is not None,
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
