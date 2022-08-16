"""Submodel dataclass module."""

from dataclasses import dataclass


from arviz import InferenceData
from arviz.data.utils import extract_dataset

from pymc.model import Model, PointFunc
from bambi.backend.pymc import PyMCModel
from pymc.util import is_transformed_name, get_untransformed_name

import xarray as xr

import numpy as np

from kulprit.families.family import Family


@dataclass
class SubModel:
    """Submodel dataclass.

    Attributes:
        idata (InferenceData): The inference data object of the submodel
            containing the posterior draws achieved by optimisation.
        backend (pymc.model.PyMCModel): The underlying PyMC backend model for the reference
            model, this is inherited by the submodels in order to retrieve
            parameter transformations
        kl_div (float): The KL divergence between the submodel and the reference
            model
        size (int): The number of common terms in the model, not including the
            intercept
        term_names (list): The names of the terms in the model, including the
            intercept
        model (bambi.Model): The underlying reference PyMC model, useful for
            understanding the structure inherited by subodels
        model_logp (pymc.model.PointFunc): The function for computing the log
            probability of the reference model. We use this same function for the
            submodel in order to perform predictive checks
        transforms (dict): A dictionary of the transforms applied by PyMC on the
            parameters in the model
        num_chain (int): The number of chains in the submodel's posterior
        num_draw (int): The number of draws in each of the submodel's chains
        num_samples (int): The number of total samples in the submodel's
            posterior
        num_obs (int): The number of data observations
    """

    model: Model
    idata: InferenceData
    loss: float
    size: int
    term_names: list
