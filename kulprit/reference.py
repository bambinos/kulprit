"""Core reference model class."""

from typing import Tuple, Union, Optional, List
from typing_extensions import Literal

import matplotlib

import arviz as az
import bambi as bmb

import pandas as pd

from kulprit.data.submodel import SubModel
from kulprit.plots.plots import plot_compare
from kulprit.projection.projector import Projector
from kulprit.search.searcher import Searcher


class ProjectionPredictive:
    """
    Projection Predictive class from which we perform the model selection procedure.
    """

    def __init__(
        self,
        model: bmb.models.Model,
        idata: Optional[az.InferenceData] = None,
    ) -> None:
        """Builder for projection predictive model selection.

        This object initializes the reference model and handles the core projection, variable search
        methods and submodel comparison.

        Parameters:
        ----------
        model : Bambi model
            The reference GLM model to project
        idata : InferenceData
            The ArviZ InferenceData object of the fitted reference model
        """
        # test that the reference model has an intercept term
        if model.response_component.intercept_term is None:
            raise UserWarning(
                "The procedure currently requires reference models to have an intercept term."
            )

        # test that the reference model does not admit any hierarchical structure
        if model.response_component.group_specific_terms:
            raise NotImplementedError("Hierarchical models currently not supported.")

        # build posterior if not provided
        if idata is None:
            idata = model.fit(idata_kwargs={"log_likelihood": True})
        elif "log_likelihood" not in idata.groups():
            raise UserWarning(
                """Please run Bambi's fit method with the option
                idata_kwargs={'log_likelihood': True}"""
            )

        # test compatibility between model and idata
        if not check_model_idata_compatability(model=model, idata=idata):
            raise UserWarning("Incompatible model and inference data.")

        # log reference model and inference data
        self.model = model
        self.idata = idata

        # instantiate projector, searcher classes
        self.projector = Projector(model=self.model, idata=self.idata)
        self.searcher = Searcher(self.projector)
        # we have not yet run a search
        self.path = None

    def __repr__(self) -> str:
        """String representation of the reference model."""

        # return the formular for the reference model if no search has been run
        if self.path is None:
            return (
                f"ReferenceModel("
                f"{', '.join([self.model.formula.main] + list(self.model.formula.additionals))})"
            )

        # otherwise return the formulas for the submodels
        else:
            str_of_submodels = "\n".join(
                f"{idx:>3} "
                f"{', '.join([value.model.formula.main] + list(value.model.formula.additionals))}"
                for idx, value in enumerate(self.path.values())
            )
            return str_of_submodels

    def project(
        self,
        terms: Union[List[str], Tuple[str], int],
    ) -> SubModel:
        """Projection the reference model onto a variable subset.

        Parameters:
        -----------
        terms : Union[List[str], Tuple[str], int]
        Collection of strings containing the names of the parameters to include the submodel,
        or the number of parameters to include in the submodel, not including the intercept term

        Returns:
        --------
        kulprit.data.SubModel: Projected submodel object
        """

        # project the reference model onto a subset of covariates
        sub_model = self.projector.project(terms=terms)
        return sub_model

    def search(
        self,
        max_terms: Optional[int] = None,
        method: Literal["forward", "l1"] = "forward",
    ) -> Optional[dict]:
        """Model search method through parameter space.

        If ``max_terms`` is not provided, then the search path runs from the intercept-only model
        up to but not including the full model.

        Parameters:
        -----------
        max_terms : int
            The number of parameters of the largest submodel in the search path, not including the
            intercept term.
        method : str
            The search method to employ, either "forward" to employ a forward search heuristic
            through the space, or "l1" to use the L1-regularized search path.

        Returns:
        --------
        dict: The model selection procedure search path, containing the submodels along the
            search path, keyed by their model size.
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

    def plot_compare(
        self,
        plot: Optional[bool] = False,
        legend: Optional[bool] = True,
        title: Optional[bool] = True,
        figsize: Optional[tuple] = None,
        plot_kwargs: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, matplotlib.axes.Axes]:
        """Compare the ELPD of the projected models along the search path.


        Parameters:
        -----------
        plot : bool
            Plot the results of the comparison. Defaults to False
        legend : bool
            Add legend to figure. Defaults to True.
        title : bool
            Show a tittle with a description of how to interpret the plot. Defaults to True.
        figsize : tuple
            If None, size is (10, num of submodels) inches
        plot_kwargs : dict
            Optional arguments for plot elements. Currently accepts 'color_elpd', 'marker_elpd',
        'marker_fc_elpd', 'color_dse', 'marker_dse', 'ls_reference', 'color_ls_reference'.

        Returns:
        --------
        cmp : DataFrame
            ordered from largest to smaller model. The columns are:

        - rank: The rank-order of the models. 0 is the best.
        - elpd: ELPD estimated either using (PSIS-LOO-CV). Higher ELPD indicates higher
            out-of-sample predictive fit ("better" model).
        - pIC: Estimated effective number of parameters.
        - elpd_diff: The difference in ELPD between two models.
            The difference is computed relative to the reference model
        - weight: Relative weight for each model. This can be loosely interpreted as the probability
            of each model (among the compared model) given the data.
        - SE: Standard error of the ELPD estimate.
        - dSE: Standard error of the difference in ELPD between each model and the top-ranked model.
            It's always 0 for the reference model.
        - warning: A value of 1 indicates that the computation of the ELPD may not be reliable.
            This could be indication of PSIS-LOO-CV starting to fail see
            http://arxiv.org/abs/1507.04544 for details.
        - scale: Scale used for the ELPD. This is always the log scale

        axes : matplotlib_axes or bokeh_figure
        """

        # test that search has been previously run
        if self.searcher.search_completed is False:
            raise UserWarning("Please run search before comparing submodels.")

        # initiate plotting arguments if none provided
        if plot_kwargs is None:
            plot_kwargs = {}

        self.searcher.idatas = {}  # pylint: disable=attribute-defined-outside-init

        # make dictionary of inferencedata objects for each projection
        for k, submodel in self.searcher.path.k_submodel.items():
            self.searcher.idatas[k] = submodel.idata
        self.searcher.idatas[
            k + 1  # pylint: disable=undefined-loop-variable
        ] = self.searcher.projector.idata

        # compare the submodels using loo (other criteria may be added in the future)
        comparison = az.compare(self.searcher.idatas)
        comparison.sort_index(ascending=False, inplace=True)

        # plot the comparison if requested
        axes = None
        if plot:
            axes = plot_compare(comparison, legend, title, figsize, **plot_kwargs)

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
                set(self.model.response_component.terms.keys()) - set([self.model.response_name])
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


def check_model_idata_compatability(model, idata):
    """Check that the Bambi model and idata are compatible with vanilla procedure.

    In the future, this will be extended to allow for different structures and
    covariates, checking instead only that the observation data are the same.

    Parameters:
    ----------
    model : Bambi model
        The reference GLM model to project
    idata : InferenceData
        The ArviZ InferenceData object of the fitted reference model

    Returns:
    -------
        bool : Indicator of whether the two objects are compatible
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
