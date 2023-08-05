"""Core search module."""

from typing import Optional
from typing_extensions import Literal

import pandas as pd
import arviz as az

from kulprit.projection.projector import Projector
from kulprit.plots.plots import plot_compare
from kulprit.search.forward import ForwardSearchPath
from kulprit.search.l1 import L1SearchPath


class Searcher:
    def __init__(self, projector: Projector) -> None:
        """Initialise forward search class.

        Parameters:
        ----------
        projector : Projector
            A projector object.

        Raises
        ------
            UserWarning: If method is not "forward" or "l1".
        """
        # initialise projector for the search procedure
        self.projector = projector

        # define all available search heuristics
        self.method_dict = {
            "forward": ForwardSearchPath,
            "l1": L1SearchPath,
        }

        # indicator variable tracking whether or not a search has been run
        self.search_completed = False

    def __repr__(self) -> str:
        """String representation of the search path."""

        return self.path.__repr__()

    def search(self, max_terms: int, method: Literal["forward", "l1"] = "forward") -> dict:
        """Primary search method of the procedure.

        Performs forward search through the parameter space.

        Parameters:
        ----------
        max_terms : int
            Number of terms to perform the forward search up to.
        method : str
            Method to use for search.

        Returns:
        -------
        dict: A dictionary of submodels, keyed by the number of terms in the submodel.
        """

        # test valid solution method
        if method not in self.method_dict:
            raise UserWarning("Please either select either forward search or L1 search.")

        # initialise search path
        self.path = self.method_dict[method](  # pylint: disable=attribute-defined-outside-init
            self.projector
        )

        # perform the search according to the chosen heuristic
        k_submodels = self.path.search(max_terms=max_terms)

        # toggle indicator variable and return search path
        self.search_completed = True

        # feed path result through to the projector
        self.projector.path = k_submodels

        # return the final search path
        return k_submodels

    def loo_compare(
        self,
        plot: Optional[bool] = False,
        legend: Optional[bool] = True,
        title: Optional[bool] = True,
        figsize: Optional[tuple] = None,
        plot_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Rank submodels by ELPD and plot the comparison.

        Parameters:
        ----------
        plot : bool
            Whether or not to plot the comparison.
        legend : bool
            Whether or not to include a legend in the plot.
        title : bool
            Whether or not to include a title in the plot.
        figsize : tuple
            Figure size for the plot.
        plot_kwargs : dict
            Keyword arguments for the plot.

        Returns:
        -------
        pd.DataFrame: A dataframe of the comparison results.
        axes: A tuple of matplotlib axes objects.
        """
        # test that search has been previously run
        if self.search_completed is False:
            raise UserWarning("Please run search before comparing submodels.")

        # initiate plotting arguments if none provided
        if plot_kwargs is None:
            plot_kwargs = {}

        self.idatas = {}  # pylint: disable=attribute-defined-outside-init

        # make dictionary of inferencedata objects for each projection
        for k, submodel in self.path.k_submodel.items():
            self.idatas[k] = submodel.idata
        self.idatas[k + 1] = self.projector.idata  # pylint: disable=undefined-loop-variable

        # compare the submodels by some criterion
        comparison = az.compare(self.idatas)
        comparison.sort_index(ascending=False, inplace=True)

        # plot the comparison if requested
        axes = None
        if plot:
            axes = plot_compare(comparison, legend, title, figsize, **plot_kwargs)

        return comparison, axes
