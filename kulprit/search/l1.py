"""L1 search path module."""

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
        terms = self.projector.model.components[self.projector.model.family.likelihood.parent].terms

        # test whether the model includes categorical terms, and if so raise error
        if sum(terms[term].categorical for term in terms.keys()) > 0:
            raise NotImplementedError("Group-lasso not yet implemented")

        # initialise search
        self.k_term_names = {}
        self.k_submodel = {}

        self.search_completed = True

    def __repr__(self) -> str:
        """String representation of the search path."""

        path_dict = {
            k: [
                list(
                    submodel.model.components[
                        submodel.model.family.likelihood.parent
                    ].common_terms.keys()
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

    def first_non_zero_idx(self, arr):
        """Find the index of the first non-zero element in each row of a matrix.

        Parameters:
        ----------

        arr : np.ndarray
            A matrix.

        Returns:
        -------
        dict: Dictionary keyed by the row number where each value is the index of the first
        non-zero element in that row.
        """

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

        We compute the L1 search path for a given data object in the latent space and return the
        coefficients for each model size. This use of the latent space is a bit of a hack,
        but Catalina et al. (2021) show that this space remains informative in terms of model
        selection and results in faster computation. We use ``sklearn`` to compute the L1 path.

        Returns:
        -------
        np.ndarray: The coefficients for each model size
        """
        model = self.projector.model
        # extract reference model data and latent predictor
        self.common_terms = list(  # pylint: disable=attribute-defined-outside-init
            model.components[model.family.likelihood.parent].common_terms
        )
        d_component = model.distributional_components[model.family.likelihood.parent]
        X = np.column_stack([d_component.design.common[term] for term in self.common_terms])
        # XXX we need to make this more general  # pylint: disable=fixme
        mean_param_name = list(self.projector.model.family.link.keys())[0]
        eta = self.projector.model.family.link[mean_param_name].link(
            model.components[model.family.likelihood.parent].design.response.design_matrix
        )
        # compute L1 path in the latent space
        _, coef_path, _ = lasso_path(X, eta)
        cov_order = self.first_non_zero_idx(coef_path)
        return cov_order

    def search(self, max_terms: int) -> dict:
        """Perform L1 search through the parameter space.

        Parameters:
        ----------
        max_terms : int
            Number of terms to perform the forward search up to.

        Returns:
        -------
        dict: A dictionary of submodels, keyed by the number of terms in the submodel.
        """

        # compute L1 path for each model size
        coef_path = self.compute_path()

        # sort the covariates according to their L1 ordering
        cov_lasso = dict(sorted(coef_path.items(), key=lambda item: item[1]))
        sorted_covs = [self.common_terms[k] for k in cov_lasso]

        # produce submodels for each model size
        self.k_term_names = {k: sorted_covs[:k] for k in range(max_terms + 1)}

        # project the reference model on each of the submodels
        for k, term_names in self.k_term_names.items():
            self.k_submodel[k] = self.projector.project(terms=term_names)

        # toggle indicator variable and return search path
        self.search_completed = True
        return self.k_submodel
