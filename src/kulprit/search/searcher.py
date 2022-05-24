"""Core search module."""

from typing import Optional
from typing_extensions import Literal

import pandas as pd
import arviz as az

from kulprit.data.data import ModelData
from kulprit.projection.projector import Projector
from kulprit.search.path import SearchPath


class Searcher:
    def __init__(self, data: ModelData, projector: Projector) -> None:
        """Initialise forward search class."""

        # initialise reference model and set of reference model term names
        self.data = data
        self.term_names = data.structure.common_terms

        # initialise projector for the search procedure
        self.projector = projector

        # initialise search path
        self.path = SearchPath(ref_terms=self.term_names)

        # indicator variable tracking whether or not a search has been run
        self.search_completed = False

    def __repr__(self) -> str:
        """String representation of the search path."""

        return self.path.__repr__()

    def search(
        self, max_terms: int, method: Literal["analytic", "gradient"]
    ) -> SearchPath:
        """Primary search method of the procedure.

        Performs forward search through the parameter space.

        Args:
            max_terms (int): Number of terms to perform the forward search
                up to.
            method (str): The projection method to employ, either "analytic" to
                use the hard-coded solutions the optimisation problem, or
                "gradient" to employ gradient descent methods
        """

        # initial intercept-only subset
        k = 0
        k_term_names = []
        k_submodel = self.projector.project(terms=k_term_names, method=method)
        k_dist = k_submodel.dist_to_ref_model

        # add submodel to search path
        self.path.add_submodel(
            k=k,
            k_term_names=k_term_names,
            k_submodel=k_submodel,
            k_dist=k_dist,
        )

        # perform forward search through parameter space
        while k < max_terms:
            # get list of candidate submodels, project onto them, and compute
            # their distances
            k_candidates = self.path.get_candidates(k=k)
            k_projections = [
                self.projector.project(terms=candidate, method=method)
                for candidate in k_candidates
            ]

            # identify the best candidate by distance from reference model
            best_submodel = min(
                k_projections, key=lambda projection: projection.sort_index
            )
            best_dist = best_submodel.sort_index

            # retrieve the best candidate's term names and indices
            k_term_names = best_submodel.structure.common_terms

            # increment number of parameters
            k = len(k_term_names)

            # add best candidate to search path
            self.path.add_submodel(
                k=k,
                k_term_names=k_term_names,
                k_submodel=best_submodel,
                k_dist=best_dist,
            )

        # toggle indicator variable and return search path
        self.search_completed = True
        return self.path

    def loo_compare(
        self,
        ic: Optional[Literal["loo", "waic"]] = None,
        method: Literal["stacking", "BB-pseudo-BMA", "pseudo-MA"] = "stacking",
        b_samples: int = 1000,
        alpha: float = 1,
        seed=None,
        scale: Optional[Literal["log", "negative_log", "deviance"]] = None,
        var_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare the ELPD of the projected models along the search path."""

        # make dictionary of inferencedata objects for each projection
        self.idatas = {k: submodel.idata for k, submodel in self.path.k_submodel.items()}

        # compare the submodels by some criterion
        comparison = az.compare(
            self.idatas,
            ic=ic,
            method=method,
            b_samples=b_samples,
            alpha=alpha,
            seed=seed,
            scale=scale,
            var_name=var_name,
        )
        return comparison
