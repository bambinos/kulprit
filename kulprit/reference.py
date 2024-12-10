"""Core reference model class."""

from typing import Tuple, Union, Optional, List
from typing_extensions import Literal

import matplotlib

import arviz as az
import bambi as bmb

import pandas as pd

from kulprit.data.submodel import SubModel
from kulprit.plots.plots import plot_compare, plot_densities
from kulprit.projection.projector import Projector
from kulprit.search.forward import ForwardSearchPath
from kulprit.search.l1 import L1SearchPath


class ProjectionPredictive:
    """
    Projection Predictive class from which we perform the model selection procedure.
    """

    def __init__(
        self,
        model: bmb.models.Model,
        idata: Optional[az.InferenceData] = None,
        num_samples: int = 400,
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
        self.has_intercept = bmb.formula.formula_has_intercept(model.formula.main)
        if not self.has_intercept:
            raise UserWarning(
                "The procedure currently requires reference models to have an intercept term."
            )

        # test that the reference model does not admit any hierarchical structure
        if any(val.group_specific_groups for val in model.distributional_components.values()):
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

        self.searcher_path = None
        # indicator variable tracking whether or not a search has been run
        self.search_completed = False

        # instantiate projector, searcher classes
        self.projector = Projector(
            model=self.model,
            idata=self.idata,
            num_samples=num_samples,
            has_intercept=self.has_intercept,
        )
        # we have not yet run a search
        self.path = None

    def __repr__(self) -> str:
        """String representation of the reference model."""

        # return the formular for the reference model if no search has been run
        if self.path is None:
            return "ReferenceModel"

        # otherwise return the formulas for the submodels
        else:
            str_of_submodels = "\n".join(
                f"{idx:>3} " f"{value}" for idx, value in enumerate(self.path.values())
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

        # test valid solution method
        if method not in ["forward", "l1"]:
            raise ValueError("Please select either forward search or L1 search.")

        # initialise search path
        if method == "forward":
            self.searcher_path = ForwardSearchPath(self.projector)
        else:
            self.searcher_path = L1SearchPath(self.projector)

        # set default `max_terms` value
        n_terms = len(self.model.components[self.model.family.likelihood.parent].common_terms)
        if max_terms is None:
            max_terms = n_terms
        # test `max_terms` input
        elif max_terms > n_terms:
            raise UserWarning(
                "Please ensure that the maximum number to consider in the "
                + "submodel search is between 1 and the number of terms in the "
                + "reference model."
            )

        # perform the search according to the chosen heuristic
        k_submodels = self.searcher_path.search(max_terms=max_terms)

        # feed path result through to the projector
        self.projector.path = k_submodels
        self.path = k_submodels
        # toggle indicator variable and return search path
        self.search_completed = True

    def compare(
        self,
        plot: Optional[bool] = True,
        min_model_size: Optional[int] = 0,
        legend: Optional[bool] = True,
        title: Optional[bool] = True,
        figsize: Optional[tuple] = None,
        plot_kwargs: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, matplotlib.axes.Axes]:
        """Compare the ELPD of the projected models along the search path.


        Parameters:
        -----------
        plot : bool
            Plot the results of the comparison. Defaults to True
        legend : bool
            Add legend to figure. Defaults to True.
        title : bool
            Show a tittle with a description of how to interpret the plot. Defaults to True.
        figsize : tuple
            If None, size is (10, num of submodels) inches
        plot_kwargs : dict
            Optional arguments for plot elements. Currently accepts 'color_elpd', 'marker_elpd',
        'marker_fc_elpd', 'color_dse', 'marker_dse', 'ls_reference', 'color_ls_reference', 
        'xlabel_rotation'.

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
        if self.search_completed is False:
            raise UserWarning("Please run search before comparing submodels.")

        # initiate plotting arguments if none provided
        if plot_kwargs is None:
            plot_kwargs = {}

        self.searcher_idatas = {}  # pylint: disable=attribute-defined-outside-init

        # make dictionary of inferencedata objects for each projection
        for k, submodel in self.searcher_path.k_submodel.items():
            if k >= min_model_size:
                self.searcher_idatas[k] = submodel.idata
        self.searcher_idatas[
            k + 1  # pylint: disable=undefined-loop-variable
        ] = self.projector.idata

        label_terms = ["Intercept"] if self.has_intercept else []
        label_terms.extend([term_name for term_name in submodel.term_names])

        # compare the submodels using loo (other criteria may be added in the future)
        comparison = az.compare(self.searcher_idatas)
        comparison.sort_index(ascending=False, inplace=True)

        # plot the comparison if requested
        axes = None
        if plot:
            axes = plot_compare(comparison, label_terms, legend, title, figsize, plot_kwargs)

        return comparison, axes

    def plot_densities(
        self,
        var_names: Optional[List[str]] = None,
        submodels: Optional[List[int]] = None,
        include_reference: bool = True,
        labels: Literal["formula", "size"] = "formula",
        kind: Literal["density", "forest"] = "density",
        figsize: Optional[Tuple[int, int]] = None,
        plot_kwargs: Optional[dict] = None,
    ) -> matplotlib.axes.Axes:
        """Compare the projected posterior densities of the submodels

        Parameters:
        -----------
        var_names : list of str, optional
            List of variables to plot.
        submodels : list of int, optional
            List of submodels to plot, 0 is intercept-only model and the largest valid integer is
            the total number of variables in reference model. If None, all submodels are plotted.
        include_reference : bool
            Whether to include the reference model in the plot. Defaults to True.
        labels : str
            If "formula", the labels are the formulas of the submodels. If "size", the number
            of covariates in the submodels.
        figsize : tuple
            Figure size. If None it will be defined automatically.
        plot_kwargs : dict
            Dictionary passed to ArviZ's ``plot_density`` function (if kind density) or to
            ``plot_forest`` (if kind forest).

        Returns:
        --------

        axes : matplotlib_axes or bokeh_figure
        """
        return plot_densities(
            self.model,
            self.path,
            self.idata,
            var_names=var_names,
            submodels=submodels,
            include_reference=include_reference,
            labels=labels,
            kind=kind,
            figsize=figsize,
            plot_kwargs=plot_kwargs,
        )


def check_model_idata_compatability(model, idata):
    """Check that the Bambi model and idata are compatible with vanilla procedure.

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
    if not model.response_component.term.name == list(idata.observed_data.data_vars.variables)[0]:
        return False

    # return default truth
    return True
