"""Core reference model class."""

from typing import Union, Optional, List

import torch
from arviz import InferenceData
from bambi.models import Model

from .data import ModelData
from .data.structure import ModelStructure
from .projection.projector import Projector
from .search.searcher import Searcher
from .search.path import SearchPath
from .formatting import spacify, multilinify


class ReferenceModel:
    """
    Reference model class from which we perform the model selection procedure.
    """

    def __init__(
        self,
        model: Model,
        idata: Optional[InferenceData] = None,
        num_iters: Optional[int] = 200,
        learning_rate: Optional[float] = 0.01,
    ) -> None:
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Note that throughout the procedure, variables with names of the form
        ``*_ast`` belong to the reference model while variables with names like
        ``*_perp`` belong to the restricted model. This is to preserve notation
        choices from previous papers on the topic.

        Args:
            model (bambi.models.Model): The referemce GLM model to project
            idata (arviz.InferenceData): The arViz InferenceData object
                of the fitted reference model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate
        """

        # build posterior if not provided
        if idata is None:
            idata = model.fit()

        # build model data class
        structure = ModelStructure(model)
        self.ref_model = ModelData(
            structure=structure, idata=idata, dist_to_ref_model=torch.tensor(0)
        )

        # instantiate projector and search class
        self.projector = Projector(
            self.ref_model, num_iters=num_iters, learning_rate=learning_rate
        )
        self.searcher = Searcher(self.ref_model)

    def project(
        self,
        terms: Union[List[str], int],
    ) -> ModelData:
        """Projection the reference model onto a variable subset.

        Args:
            terms (Union[List[str], int]): Either a list of strings containing
                the names of the parameters to include the submodel, or the
                number of parameters to include in the submodel, **not**
                including the intercept term

        Returns:
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # Todo:
        # 1. add tests on `terms`

        # project the reference model onto a subset of covariates
        sub_model = self.projector.project(terms)
        return sub_model

    def search(
        self,
        max_terms: Optional[int] = None,
        num_iters: Optional[int] = 200,
        learning_rate: Optional[float] = 0.01,
    ) -> SearchPath:
        """Model search method through parameter space.

        If ``max_terms`` is not provided, then the search path runs from the
        intercept-only model up to but not including the full model, in other
        words we set ``max_terms = ref_model.num_terms - 1``.

        Args:
            max_terms (int): The number of parameters of the largest submodel in
                the search path, **not** including the intercept term
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate

        Returns:
            kulprit.search.SearchPath: The model selection procedure search path
        """

        # Todo:
        # 1. add tests on `max_terms`

        if max_terms is None:
            max_terms = self.ModelData.num_terms - 1

        raise NotImplementedError(
            "This method is still in development, sorry about that!"
        )
