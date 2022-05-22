"""Search path module."""

from typing import List

import pandas as pd

from ..data import ModelData


class SearchPath:
    """Search path object for model selection procedure."""

    def __init__(self, ref_terms: list) -> None:
        """Initialise search path class."""

        # set limit to number of terms in search path
        self.ref_terms = ref_terms

        # initialise search
        self.k_term_idx = {}
        self.k_term_names = {}
        self.k_submodel = {}
        self.k_dist = {}

    def __getitem__(self, k: int) -> ModelData:
        """Return the submodel in the search path with k terms."""

        return self.k_submodel[k]

    def __repr__(self) -> str:
        """String representation of the search path."""

        path_dict = {
            k: [submodel.structure.term_names, submodel.dist_to_ref_model]
            for k, submodel in self.k_submodel.items()
        }
        df = pd.DataFrame.from_dict(
            path_dict, orient="index", columns=["Terms", "Distance from reference model"]
        )
        df.index.name = "Model size"
        string = df.to_string()
        return string

    def add_submodel(
        self,
        k: int,
        k_term_names: list,
        k_submodel: ModelData,
        k_dist: float,
    ) -> None:
        """Update search path with new submodel."""

        self.k_term_names[k] = k_term_names
        self.k_submodel[k] = k_submodel
        self.k_dist[k] = k_dist

    def get_candidates(self, k: int) -> List[List]:
        """Method for extracting a list of all candidate submodels.

        Args:
            k (int): The number of terms in the previous submodel, from which we
            wish to
                find all possible candidate submodels

        Returns:
            List[List]: A list of lists, each containing the terms of all candidate
            submodels
        """

        prev_subset = self.k_term_names[k]
        candidate_additions = list(set(self.ref_terms).difference(prev_subset))
        candidates = [prev_subset + [addition] for addition in candidate_additions]
        return candidates
