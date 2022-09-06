"""Core reference model class."""

from typing import Tuple, Union, Optional, List
from typing_extensions import Literal

import matplotlib

import arviz as az
import bambi as bmb

import pandas as pd

from kulprit.data.submodel import SubModel
from kulprit.projection.projector import Projector
from kulprit.search.searcher import Searcher


class ReferenceModel:
    """
    Reference model class from which we perform the model selection procedure.
    """

    def __init__(
        self,
        model: bmb.models.Model,
        idata: Optional[az.InferenceData] = None,
    ) -> None:
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Note that throughout the procedure, variables with names of the form
        ``*_ast`` belong to the reference model while variables with names like
        ``*_perp`` belong to the restricted model. This is to preserve notation
        choices from previous papers on the topic.

        Args:
            model (bmb.models.Model): The referemce GLM model to project
            idata (az.InferenceData): The ArviZ InferenceData object
                of the fitted reference model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate
            num_thinned_samples (int): The number of draws to use in optimisation
        """

        # test that the reference model has an intercept term
        if model.intercept_term is None:
            raise UserWarning(
                "The procedure currently requires reference models to have an"
                + " intercept term."
            )

        # test that the reference model does not admit any hierarchical structure
        if model.group_specific_terms:
            raise NotImplementedError("Hierarchical models currently not supported.")

        # build posterior if not provided
        if idata is None:
            idata = model.fit()

        # test compatibility between model and idata
        if not test_model_idata_compatability(model=model, idata=idata):
            raise UserWarning("Incompatible model and inference data.")

        # log reference model and inference data
        self.model = model
        self.idata = idata

        # instantiate projector, search, and search path classes
        self.projector = Projector(model=self.model, idata=self.idata)
        self.searcher = Searcher(self.projector)
        self.path = None

    def project(
        self,
        terms: Union[List[str], Tuple[str], int],
    ) -> SubModel:
        """Projection the reference model onto a variable subset.

        Args:
            terms (Union[List[str], Tuple[str], int]): Collection of strings
                containing the names of the parameters to include the submodel,
                or the number of parameters to include in the submodel, **not**
                including the intercept term

        Returns:
            kulprit.data.SubModel: Projected submodel object
        """

        # project the reference model onto a subset of covariates
        sub_model = self.projector.project(terms=terms)
        return sub_model

    def search(
        self,
        max_terms: Optional[int] = None,
        method: Literal["forward", "l1"] = "forward",
        return_path: Optional[bool] = False,
    ) -> Optional[dict]:
        """Model search method through parameter space.

        If ``max_terms`` is not provided, then the search path runs from the
        intercept-only model up to but not including the full model, in other
        words we set ``max_terms = data.num_terms - 1``.

        Args:
            max_terms (int): The number of parameters of the largest submodel in
                the search path, **not** including the intercept term
            method (str): The search method to employ, either "forward" to
                employ a forward search heuristic through the space, or "l1" to
                use the L1-regularised search path
            return_path (bool): Whether or not to return the search path as a
                dictionary object to the user

        Returns:
            dict: The model selection procedure search path, containing the
                submodels along the search path, keyed by their model size
        """

        # set default `max_terms` value
        if max_terms is None:
            max_terms = len(self.model.common_terms)

        # test `max_terms` input
        if max_terms > len(self.model.common_terms):
            raise UserWarning(
                "Please ensure that the maximum number to consider in the "
                + "submodel search is between 1 and the number of terms in the "
                + "reference model."
            )

        # perform search through the parameter space
        self.path = self.searcher.search(
            max_terms=max_terms,
            method=method,
        )

        # return the path dictionary if specified
        if return_path:  # pragma: no cover
            return self.path

    def loo_compare(
        self,
        ic: Optional[Literal["loo", "waic"]] = None,
        plot: Optional[bool] = False,
        method: Literal["stacking", "BB-pseudo-BMA", "pseudo-MA"] = "stacking",
        b_samples: int = 1000,
        alpha: float = 1,
        seed=None,
        scale: Optional[Literal["log", "negative_log", "deviance"]] = None,
        var_name: Optional[str] = None,
        plot_kwargs: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, matplotlib.axes.Axes]:

        # perform pair-wise predictive performance comparison with LOO
        comparison, axes = self.searcher.loo_compare(
            ic=ic,
            plot=plot,
            method=method,
            b_samples=b_samples,
            alpha=alpha,
            seed=seed,
            scale=scale,
            var_name=var_name,
            plot_kwargs=plot_kwargs,
        )
        return comparison, axes

    def plot_densities(
        self,
        var_names: Optional[List[str]] = None,
        outline: Optional[bool] = False,
        shade: Optional[float] = 0.4,
    ) -> matplotlib.axes.Axes:
        """Compare the projected posterior densities of the submodels"""

        # set default variable names to the reference model terms
        if not var_names:
            var_names = self.model.term_names

        axes = az.plot_density(
            data=[submodel.idata for submodel in self.path.values()],
            group="posterior",
            var_names=var_names,
            outline=outline,
            shade=shade,
            data_labels=[submodel.model.formula for submodel in self.path.values()],
        )
        return axes


def test_model_idata_compatability(model, idata):
    """Test that the Bambi model and idata are compatible with vanilla procedure.

    In the future, this will be extended to allow for different structures and
    covariates, testing instead only that the observation data are the same.

    Args:
        model (bmb.models.Model): The reference Bambi model object
        idata (az.InferenceData): The reference model fitted inference data obejct

    Returns:
        bool: Indicator of whether the two objects are compatible
    """

    # test that the variate's name is the same in reference model and idata
    if not model.response.name == list(idata.observed_data.data_vars.variables)[0]:
        return False

    # test that the variate has the same dimensions in reference model and idata
    if not (
        idata.observed_data[model.response.name].to_numpy().shape
        == model.data[model.response.name].shape
    ):
        return False

    # return default truth
    return True
