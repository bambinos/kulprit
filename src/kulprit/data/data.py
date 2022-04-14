"""Module for handling data throughout the procedure."""

import dataclasses

import pandas

import torch

import arviz
import pymc3
import formulae
import bambi
import kulprit

from ..families import Family


@dataclasses.dataclass(order=True)
class ModelData:
    """Data class for handling model data.

    This class serves as the primary data container passed throughout the
    procedure, allowing for more simple and legible code. Note that this class
    supports ordering, and we choose distance to reference model as our sorting
    index. Naturally, this value is set to zero for the reference model,
    providing a hard minimum value.

    Attributes:
        X (torch.tensor): Model design matrix
        y (torch.tensor): Model variate observations
        backend (pymc3.Model): The PyMC3 model backend
        design (formulae.matrices.DesignMatrices): The formulae design matrix
            object underpinning the GLM
        link (bambi.families.Link): GLM link function object
        family (kulprit.families.Family): Model variate family object
        term_names (list): List of model covariates in their order of appearance
            **not** including the `Intercept` term
        common_terms (list): List of all terms in the model in order of
            appearance (includes the `Intercept` term)
        response_name (str): The name of the response given to the Bambi model
        num_obs (int): Number of data observations
        num_terms (int): Number of variables observed, and equivalently the
            number of common terms in the model (including intercept)
        num_draws (int): Number of posterior draws in the model
        model_size (int): Number of common terms in the model (terms not
            including the intercept)
        has_intercept (bool): Flag whether intercept included in model
        dist_to_ref_model (torch.tensor): The Kullback-Leibler divergence
            between this model and the reference model
        inferencedata (arviz.InferenceData): InferenceData object of the model
        predictions (arviz.InferenceData): In-sample model predictions
        elpd (arviz.ELPDData): Model ELPD LOO estimates
        sort_index (int): Sorting index attribute used in forward search method
    """

    X: torch.tensor
    y: torch.tensor
    backend: pymc3.Model
    design: formulae.matrices.DesignMatrices
    link: bambi.families.Link
    family: kulprit.families.Family
    term_names: list
    common_terms: list
    response_name: str
    num_obs: int
    num_terms: int
    num_draws: int
    model_size: int
    has_intercept: bool
    dist_to_ref_model: torch.tensor
    inferencedata: arviz.InferenceData = None
    predictions: arviz.InferenceData = None
    elpd: arviz.ELPDData = None
    sort_index: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.sort_index = self.dist_to_ref_model
