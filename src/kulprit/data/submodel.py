"""Submodel factory class."""

from abc import ABC, abstractmethod
from copy import copy

from typing import Optional, List
from pymc import Model

import arviz as az
from arviz import InferenceData
from arviz.utils import one_de

import numpy as np
import torch

from kulprit.data.data import ModelData
from kulprit.data.structure import ModelStructure


class SubModel(ABC):
    """Abstract base class for submodel data classes."""

    @abstractmethod
    def create(self):  # pragma: no cover
        pass


class SubModelStructure(SubModel):
    def __init__(self, data: ModelData) -> None:
        """Submodel object used to create submodels from a reference model.

        Args:
            data (kulprit.data.ModelData): The reference model from which
                to build the submodel
        """

        # log reference model ModelData
        self.data = data

    def generate(self, var_names: List[str]) -> ModelStructure:
        """Generate new ModelStructure class attributes for a submodel.

        Args:
            var_names (list): The names of the parameters to use in the r
                estricted model

        Returns:
            Tuple: Structure attributes of the resulting submodel
        """

        if len(var_names) > 0:
            # extract the submatrix from the reference model's design matrix
            X_res = torch.from_numpy(
                np.column_stack(
                    [self.data.structure.design.common[term] for term in var_names]
                )
            ).float()
            # manually add intercept to new design matrix
            X_res = torch.hstack((torch.ones(self.data.structure.num_obs, 1), X_res))
        else:
            # intercept-only model
            X_res = torch.ones(self.data.structure.num_obs, 1).float()

        # update common term names and dimensions and build new ModelData object
        _, num_terms = X_res.shape
        submode_term_names = ["Intercept"] + var_names
        model_size = len(var_names)

        return X_res, num_terms, model_size, submode_term_names, var_names

    def create(self, var_names: List[str]) -> ModelStructure:
        """Build a submodel from a reference model containing specific terms.

        Args:
            var_names (list): The names of the parameters to use in the r
                estricted model

        Returns:
            ModelData: The resulting submodel `ModelData` object
        """

        # test that the submodel is indeed a submodel
        full_set = set(self.data.structure.term_names)
        projection_set = set(var_names)
        if not projection_set.issubset(full_set):
            raise UserWarning(
                "Please ensure that the submodel you wish to build contains "
                + "only terms from the larger model."
            )

        # copy and instantiate new ModelStructure object
        submodel_structure = copy(self.data.structure)
        (
            submodel_structure.X,
            submodel_structure.num_terms,
            submodel_structure.model_size,
            submodel_structure.term_names,
            submodel_structure.common_terms,
        ) = self.generate(var_names)

        # ensure correct dimensions
        assert submodel_structure.X.shape == (
            self.data.structure.num_obs,
            submodel_structure.model_size + 1,
        )
        return submodel_structure


class SubModelInferenceData(SubModel):
    def __init__(self, data: ModelData) -> None:
        """Submodel object used to create submodels from a reference model.

        Args:
            data (kulprit.data.ModelData): The reference model ModelData object
                from which to build the submodel
        """

        # log reference model ModelData
        self.data = data

    def create(
        self,
        submodel_structure: SubModelStructure,
        theta_perp: torch.tensor,
        disp_perp: Optional[torch.tensor] = None,
    ) -> InferenceData:
        """Convert some set of pytorch tensors into an ArviZ idata object.

        Args:
            submodel_structure (kulprit.data.SubModelStructure): The restricted
                model's structure object
            theta_perp (torch.tensor): Restricted parameter posterior projections,
                including the intercept term
            disp_perp (torch.tensor): Restricted model dispersions parameter
                posterior projections

        Returns:
            arviz.inferencedata: Restricted model idata object
        """

        # reshape `theta_perp` so it has the same shape as the reference model
        num_chain = len(self.data.idata.posterior.coords.get("chain"))
        num_draw = int(submodel_structure.num_thinned_draws / num_chain)
        num_terms = submodel_structure.num_terms
        num_obs = self.data.structure.num_obs

        theta_perp = torch.reshape(theta_perp, (num_chain, num_draw, num_terms))

        transforms = self.data.structure.transforms

        # build posterior dictionary from projected parameters
        posterior = {
            term: theta_perp[:, :, i]
            for i, term in enumerate(submodel_structure.term_names)
        }
        posterior_ = posterior.copy()
        if disp_perp is not None:
            # reshape `disp_perp` if present
            disp_perp = torch.reshape(disp_perp, (num_chain, num_draw))
            # update the posterior draws dictionary with dispersion parameter
            response_name = f"{self.data.structure.response_name}_sigma"
            disp_dict = {response_name: disp_perp}
            posterior.update(disp_dict)
            # check for transformed variables
            transform_name, transform_function = transforms[response_name]
            if transform_name:
                disp_dict = {
                    f"{response_name}_{transform_name}__": transform_function(
                        disp_perp.numpy()
                    ).eval(),
                }
                posterior_.update(disp_dict)

        # build points data from the posterior dictionaries
        points = self.posterior_to_points(posterior_)

        # compute log-likelihood of projected model from this posterior
        log_likelihood = self.compute_log_likelihood(self.data.structure.backend, points)

        # reshape the log-likelihood values to be inline with reference model
        log_likelihood.update(
            (key, value.reshape(num_chain, num_draw, num_obs))
            for key, value in log_likelihood.items()
        )

        # add observed data component of projected idata
        observed_data = {
            self.data.structure.response_name: self.data.idata.observed_data.get(
                self.data.structure.response_name
            )
            .to_dict()
            .get("data")
        }

        # build idata object for the projected model
        idata = az.data.from_dict(
            posterior=posterior,
            log_likelihood=log_likelihood,
            observed_data=observed_data,
        )
        return idata

    def posterior_to_points(self, posterior: dict) -> list:
        """Convert the posterior samples from a restricted model into list of dicts.

        This list of dicts datatype is referred to a `points` in PyMC, and is needed
        to be able to compute the log-likelihood of a projected model, here
        `res_model`.

        Args:
            posterior (dict): Dictionary of posterior restricted model samples

        Returns:
            list: The list of dictionaries of point samples
        """
        initial_point = self.data.structure.backend.model.initial_point(seed=None)

        # build samples dictionary from posterior of idata
        samples = {
            key: (
                posterior[key].flatten()
                if key in posterior.keys()
                else np.zeros((self.data.structure.num_thinned_draws,))
            )
            for key in initial_point.keys()
        }
        shapes = [val.shape for val in initial_point.values()]
        # extract observed and unobserved RV names and sample matrix
        var_names = list(samples.keys())
        obs_matrix = np.vstack(list(samples.values()))

        # build points list of dictionaries
        points = [
            {
                var_names[j]: np.full(shape, obs_matrix[j, i])
                for j, shape in zip(range(obs_matrix.shape[0]), shapes)
            }
            for i in range(obs_matrix.shape[1])
        ]

        return points

    def compute_log_likelihood(self, backend: Model, points: list) -> dict:
        """Compute log-likelihood of some data points given a PyMC model.

        Args:
            backend (pymc.Model) : PyMC model for which to compute log-likelihood
            points (list) : List of dictionaries, where each dictionary is a named
                sample of all parameters in the model

        Returns:
            dict: Dictionary of log-likelihoods at each point
        """
        log_likelihood_dict = {
            var.name: np.array(
                [self.data.structure.model_logp(point) for point in points]
            )
            for var in backend.model.observed_RVs
        }

        return log_likelihood_dict
