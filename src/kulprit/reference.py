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
        if model.response_component.intercept_term is None:
            raise UserWarning(
                "The procedure currently requires reference models to have an"
                + " intercept term."
            )

        # test that the reference model does not admit any hierarchical structure
        if model.response_component.group_specific_terms:
            raise NotImplementedError("Hierarchical models currently not supported.")

        # build posterior if not provided
        if idata is None:
            idata = model.fit(idata_kwargs={"log_likelihood": True})

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

        # test if reference model idata includes the log-likelihood
        if "log_likelihood" in idata.groups():
            del idata.log_likelihood
        # project reference model onto itself
        ref_log_likelihood = self.projector.compute_model_log_likelihood(
            model=self.model, idata=self.idata
        )
        self.idata.add_groups(
            log_likelihood={self.model.response_name: ref_log_likelihood},
            dims={self.model.response_name: [f"{self.model.response_name}_dim_0"]},
        )

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
            max_terms = len(self.model.response_component.common_terms)

        # test `max_terms` input
        if max_terms > len(self.model.response_component.common_terms):
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
        plot: Optional[bool] = False,
        legend: Optional[bool] = True,
        title: Optional[bool] = True,
        figsize: Optional[tuple] = None,
        plot_kwargs: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, matplotlib.axes.Axes]:
        """Compare the ELPD of the projected models along the search path.

        Args:
            plot : bool, default False
                Plot the results of the comparison.
            legend : bool, default True
                Add legend to figure.
            title : bool
                Show a tittle with a description of how to interpret the plot.
                Defaults to True.
            figsize : (float, float), optional
                If None, size is (10, num of submodels) inches
            plot_kwargs : dict, optional
                Optional arguments for plot elements. Currently accepts 'color_elpd',
                'marker_elpd', 'marker_fc_elpd', 'color_dse' 'marker_dse',
                'ls_reference' 'color_ls_reference'.

        Returns:
            A DataFrame, ordered from largest to smaller model. The columns are:
                rank: The rank-order of the models. 0 is the best.
                elpd: ELPD estimated either using (PSIS-LOO-CV).
                    Higher ELPD indicates higher out-of-sample predictive fit
                    ("better" model).
                pIC: Estimated effective number of parameters.
                elpd_diff: The difference in ELPD between two models.
                    The difference is computed relative to the reference model
                weight: Relative weight for each model.
                    This can be loosely interpreted as the probability of each model
                    (among the compared model) given the data.
                SE: Standard error of the ELPD estimate.
                dSE: Standard error of the difference in ELPD between each model and
                    the top-ranked model. It's always 0 for the reference model.
                warning: A value of 1 indicates that the computation of the ELPD may
                    not be reliable. This could be indication of PSIS-LOO-CV starting
                    to fail see http://arxiv.org/abs/1507.04544 for details.
                scale: Scale used for the ELPD. This is always the log scale
            axes : matplotlib_axes or bokeh_figure
        """
        comparison, axes = self.searcher.loo_compare(
            plot=plot,
            legend=legend,
            title=title,
            figsize=figsize,
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
            var_names = list(
                set(self.model.response_component.terms.keys())
                - set(self.model.response_name)
            )

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
    if not model.response_name == list(idata.observed_data.data_vars.variables)[0]:
        return False

    # test that the variate has the same dimensions in reference model and idata
    if model.response_component.design.response.kind != "proportion" and (
        idata.observed_data[model.response_name].to_numpy().shape
        != model.data[model.response_name].shape
    ):
        return False

    if model.response_component.design.response.kind == "proportion" and (
        idata.observed_data[model.response_name].to_numpy().shape
        != model.response_component.design.response.evaluate_new_data(model.data).shape
    ):
        return False

    # return default truth
    return True
