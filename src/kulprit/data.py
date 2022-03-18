"""Module for handling data throughout the procedure."""

import dataclasses

import pandas
import torch
import arviz

import bambi
import kulprit

from .families import Family


@dataclasses.dataclass(order=True)
class ModelData:
    """Data class for handling model data.

    This class serves as the primary data container passed throughout the
    procedure, allowing for more simple and legible code.

    Attributes:
        X (torch.tensor): Model design matrix
        y (torch.tensor): Model variate observations
        data (pandas.DataFrame): The dataframe used in the model
        link (bambi.families.Link): GLM link function object
        family (kulprit.families.Family): Model variate family object
        cov_names (list): List of model covariates in their order of appearance
        response_name (str): The name of the response given to the Bambi model
        n (int): Number of data observations
        m (int): Number of variables observed (including intercept)
        s (int): Number of posterior draws in the model
        has_intercept (bool): Flag whether intercept included in model
        dist_to_ref_model (torch.tensor): The Kullback-Leibler divergence
            between this model and the reference model
        posterior (arviz.InferenceData): Posterior draws from the model
        predictions (arviz.InferenceData): In-sample model predictions
        elpd (arviz.ELPDData): Model ELPD LOO estimates
        sort_index (int): Sorting index attribute used in forward search method
    """

    X: torch.tensor
    y: torch.tensor
    data: pandas.DataFrame
    link: bambi.families.Link
    family: kulprit.families.family.Family
    cov_names: list
    response_name: str
    n: int
    m: int
    s: int
    has_intercept: bool
    dist_to_ref_model: torch.tensor
    posterior: arviz.InferenceData = None
    predictions: arviz.InferenceData = None
    elpd: arviz.ELPDData = None
    sort_index: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.sort_index = self.dist_to_ref_model
