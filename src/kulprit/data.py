"""Module for handling data throughout the procedure."""

import dataclasses

import pandas
import torch
import arviz

import bambi

from .families import Family


@dataclasses.dataclass
class _Prit:
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
        n (int): Number of data observations
        m (int): Number of variables observed (including intercept)
        s (int): Number of posterior draws in the model
        has_intercept (bool): Flag whether intercept included in model
        posterior (arviz.InferenceData): Posterior draws from the model
        predictions (arviz.InferenceData): In-sample model predictions
    """

    X: torch.tensor
    y: torch.tensor
    data: pandas.DataFrame
    link: bambi.families.Link
    family: Family
    cov_names: list
    n: int
    m: int
    s: int
    has_intercept: bool
    posterior: arviz.InferenceData = None
    predictions: arviz.InferenceData = None
