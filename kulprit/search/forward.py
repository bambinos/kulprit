"""Forward search module."""

from typing import List

import pandas as pd
from kulprit.data.submodel import SubModel
from kulprit.search import SearchPath


class ForwardSearchPath(SearchPath):
    def __init__(self, projector) -> None:
        """Initialise search path class."""

        # log the projector object
        self.projector = projector

        # log the names of the terms in the reference model

        model = self.projector.model
        self.ref_terms = list(model.components[model.family.likelihood.parent].terms.keys())
        self.ref_terms.remove("Intercept")

        # initialise search
        self.k_term_idx = {}
        self.k_term_names = {}
        self.k_submodel = {}
        self.k_loss = {}
        self.search_completed = False

    def __repr__(self) -> str:
        """String representation of the search path."""

        path_dict = {
            k: [
                list(
                    submodel.model.components[submodel.model.family.likelihood.parent].terms.keys()
                ),
                submodel.loss,
            ]
            for k, submodel in self.k_submodel.items()
        }
        df = pd.DataFrame.from_dict(
            path_dict, orient="index", columns=["Terms", "Distance from reference model"]
        )
        df.index.name = "Model size"
        return repr(df)

    def add_submodel(
        self,
        k: int,
        k_term_names: list,
        k_submodel: SubModel,
        k_loss: float,
    ) -> None:
        """Update search path with new submodel."""

        self.k_term_names[k] = k_term_names
        self.k_submodel[k] = k_submodel
        self.k_loss[k] = k_loss

    def get_candidates(self, k: int) -> List[List]:
        """Method for extracting a list of all candidate submodels.

        Parameters:
        ----------
            k : int
        The number of terms in the previous submodel, from which we wish to find all
        possible candidate submodels.

        Returns:
        -------
            List: A list of lists, each containing the terms of all candidate submodels
        """

        prev_subset = self.k_term_names[k - 1]
        candidate_additions = list(set(self.ref_terms).difference(prev_subset))
        candidates = [prev_subset + [addition] for addition in candidate_additions]
        return candidates

    def search(
        self,
        max_terms: int,
    ) -> dict:
        """Forward search through the parameter space.

        Parameters:
        ----------
            max_terms : int
            Number of terms to perform the forward search up to.

        Returns:
        -------
            dict: A dictionary of submodels, keyed by the number of terms in the submodel.
        """

        # initial intercept-only subset
        k = 0
        k_term_names = []
        k_submodel = self.projector.project(terms=k_term_names)
        k_loss = k_submodel.loss
        # add submodel to search path
        self.add_submodel(
            k=k,
            k_term_names=k_term_names,
            k_submodel=k_submodel,
            k_loss=k_loss,
        )

        # perform forward search through parameter space
        while k < max_terms:
            # increment submodel size
            k += 1

            # get list of candidate submodels, project onto them, and compute
            # their distances
            k_candidates = self.get_candidates(k=k)
            k_projections = [self.projector.project(terms=candidate) for candidate in k_candidates]

            # identify the best candidate by loss (equivalent to KL min)
            best_submodel = min(k_projections, key=lambda projection: projection.loss)
            best_loss = best_submodel.loss
            import numpy as np

            if not np.isfinite(best_loss):
                continue

            # retrieve the best candidate's term names and indices
            k_term_names = best_submodel.term_names

            # add best candidate to search path
            self.add_submodel(
                k=k,
                k_term_names=k_term_names,
                k_submodel=best_submodel,
                k_loss=best_loss,
            )

        # toggle indicator variable and return search path
        self.search_completed = True
        return self.k_submodel
