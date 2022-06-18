"""L1 search path module."""

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from kulprit.data.data import ModelData
from kulprit.projection.projector import Projector
from kulprit.search import SearchPath


class L1SearchPath(SearchPath):
    """L1 search path class."""

    def __init__(self, projector: Projector) -> None:
        """Initialise L1 search path class."""

        # log the projector object
        self.projector = projector

        # log the model data object of the reference model
        self.data = projector.data

        # initialise search
        self.k_term_names = {}
        self.k_submodel = {}
        self.k_dist = {}

        self.search_completed = True

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
        k_submodel: ModelData,
        k_dist: float,
    ) -> None:
        """Update search path with new submodel."""

        self.k_submodel[k] = k_submodel
        self.k_dist[k] = k_dist

    def first_non_zero_idx(self, arr):
        """Find the index of the first non-zero element in each row of a matrix.

        Args:
            arr (np.ndarray): A matrix.

        Returns:
            dict: Dictionary keyed by the row number where each value is the index
                of the first non-zero element in that row."""

        idx_dict = {}
        for i, j in zip(*np.where(arr > 0)):
            if i in idx_dict:
                continue
            else:
                idx_dict[i] = j
        return idx_dict

    def compute_path(self) -> None:
        """Compute the L1 search path.

        We compute the L1 search path for a given data object in the latent space
        and return the coefficients for each model size. This use of the latent
        space is a bit of a hack, but Catalina et al. (2021) show that this
        space remains informative in terms of model selection and results in
        faster computation. We use ``sklearn`` to compute the L1 path.

        Returns:
            np.ndarray: The coefficients for each model size
        """

        # extract reference model data and latent predictor
        X = np.column_stack(
            [
                self.data.structure.design.common[term]
                for term in self.data.structure.common_terms
            ]
        )
        eta = self.data.structure.link.link(self.data.structure.y.numpy())

        # compute L1 path in the latent space
        _, coef_path, _ = lasso_path(X, eta)
        cov_order = self.first_non_zero_idx(coef_path)

        return cov_order

    def search(self, max_terms: int) -> None:
        """Perform L1 search through the parameter."""

        # compute L1 path for each model size
        coef_path = self.compute_path()

        # sort the covariates according to their L1 ordering
        cov_lasso = {
            k: v for k, v in sorted(coef_path.items(), key=lambda item: item[1])
        }
        sorted_covs = [self.data.structure.common_terms[k] for k in cov_lasso]
        sorted_covs = [["Intercept"] + sorted_covs[:i] for i in range(max_terms + 1)]

        # produce submodels for each model size
        self.k_term_names = {len(terms) - 1: terms for terms in sorted_covs}

        # project the reference model on each of the submodels
        for k, term_names in self.k_term_names.items():
            self.k_submodel[k] = self.projector.project(term_names)
            self.k_dist[k] = self.k_submodel[k].dist_to_ref_model

        # toggle indicator variable and return search path
        self.search_completed = True
        return self.k_submodel
