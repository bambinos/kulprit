"""L1 search path module."""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from kulprit.projection.projector import Projector
from kulprit.search import SearchPath


class L1SearchPath(SearchPath):
    """L1 search path class."""

    def __init__(self, projector: Projector) -> None:
        """Initialise L1 search path class."""

        # log the projector object
        self.projector = projector

        # test whether the model includes categorical terms, and if so raise error
        if (
            sum(
                [
                    self.projector.model.terms[term].categorical
                    for term in self.projector.model.terms.keys()
                ]
            )
            > 0
        ):
            raise NotImplementedError("Group-lasso not yet implemented")

        # initialise search
        self.k_term_names = {}
        self.k_submodel = {}
        self.k_elbo = {}

        self.search_completed = True

    def __repr__(self) -> str:
        """String representation of the search path."""

        path_dict = {
            k: [submodel.term_names, submodel.elbo]
            for k, submodel in self.k_submodel.items()
        }
        df = pd.DataFrame.from_dict(
            path_dict, orient="index", columns=["Terms", "Distance from reference model"]
        )
        df.index.name = "Model size"
        string = df.to_string()
        return string

    def first_non_zero_idx(self, arr):
        """Find the index of the first non-zero element in each row of a matrix.

        Args:
            arr (np.ndarray): A matrix.

        Returns:
            dict: Dictionary keyed by the row number where each value is the index
                of the first non-zero element in that row."""

        # initialise dictionary of indices
        idx_dict = {}

        # loop through each row and find first non-zero element
        for i, j in zip(*np.where(arr != 0)):
            if i in idx_dict:
                continue
            idx_dict[i] = j

        # identify which keys are missing and set their values to infinity
        if len(idx_dict) < arr.shape[0]:
            missing_keys = set(range(arr.shape[0])) - set(idx_dict.keys())
            for key in missing_keys:
                idx_dict[key] = np.inf

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
        self.common_terms = list(self.projector.model.common_terms)
        X = np.column_stack(
            [self.projector.model._design.common[term] for term in self.common_terms]
        )
        eta = self.projector.model.family.link.link(
            np.array(self.projector.model._design.response)
        )

        # compute L1 path in the latent space
        _, coef_path, _ = lasso_path(X, eta)
        cov_order = self.first_non_zero_idx(coef_path)
        return cov_order

    def search(
        self,
        max_terms: int,
        num_steps_search: Optional[int] = 5_000,
        obj_n_mc_search: Optional[float] = 10,
        num_steps_pred: Optional[int] = 100,
        obj_n_mc_pred: Optional[float] = 1,
    ) -> dict:
        """Perform L1 search through the parameter space."""

        # compute L1 path for each model size
        coef_path = self.compute_path()

        # sort the covariates according to their L1 ordering
        cov_lasso = {
            k: v for k, v in sorted(coef_path.items(), key=lambda item: item[1])
        }
        sorted_covs = [self.common_terms[k] for k in cov_lasso]
        sorted_covs = [["1"] + sorted_covs[:i] for i in range(max_terms)]

        # produce submodels for each model size
        self.k_term_names = {len(terms) - 1: terms for terms in sorted_covs}

        # project the reference model on each of the submodels
        for k, term_names in self.k_term_names.items():
            self.k_submodel[k] = self.projector.project(
                terms=term_names, num_steps=num_steps_pred, obj_n_mc=obj_n_mc_pred
            )
            self.k_elbo[k] = self.k_submodel[k].elbo

        # toggle indicator variable and return search path
        self.search_completed = True
        return self.k_submodel
