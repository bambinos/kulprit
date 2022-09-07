"""Base projection class."""

import copy
from typing import Optional, List, Union, Sequence
import collections

import arviz as az
import bambi as bmb
import xarray as xr
from xarray_einstats.stats import XrContinuousRV

from scipy import stats

import numpy as np

from kulprit.data.submodel import SubModel
from kulprit.projection.likelihood import LIKELIHOODS
from kulprit.projection.solver import Solver


class Projector:
    def __init__(
        self,
        model: bmb.Model,
        idata: az.InferenceData,
        path: Optional[dict] = None,
    ) -> None:
        """Reference model builder for projection predictive model selection.

        This class handles the core projection methods of the model selection
        procedure. Note that throughout the procedure, variables with names of
        the form ``*_ast`` belong to the reference model while variables with
        names like ``*_perp`` belong to the restricted model. This is to
        preserve notation choices from previous papers on the topic.

        Args:
            data (kulprit.data.ModelData): Reference model dataclass object
            path (dict): An optional search path dictionary, initialised to None
                and assigned by the ReferenceModel parent object following a
                search for efficient submodel retrieval
            num_steps (int): Number of iterations to run VI for
            obj_n_mc (int):
        """

        # log reference model and reference inference data object
        self.model = model
        self.idata = idata

        # log properties of the reference model
        self.response_name = self.model.response.name
        self.ref_family = self.model.family.name
        self.priors = self.model.family.likelihood.priors

        # build solver
        self.solver = Solver(model=self.model, idata=self.idata)

        # log search path
        self.path = path

    def project(
        self,
        terms: Union[Sequence[str], int],
    ) -> SubModel:
        """Wrapper function for projection method.

        Args:
            terms (Union[Sequence[str], int]): Collection of strings containing
            the names of the parameters to include the submodel, or the number
            of parameters to include in the submodel, **not** including the
            intercept term

        Returns:
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # project terms by name
        if isinstance(terms, collections.Sequence):
            # if not a list, cast to list
            if not isinstance(terms, list):
                terms = list(terms)

            # test `terms` input
            if not set(terms).issubset(set(self.model.common_terms)):
                raise UserWarning(
                    "Please ensure that all terms selected for projection exist in"
                    + " the reference model."
                )
            # perform projection
            return self.project_names(term_names=terms)

        # project a number of terms
        elif isinstance(terms, int):
            # test `model_size` input
            if self.path is None or terms not in list(self.path):
                raise UserWarning(
                    "In order to project onto an integer number of terms, please "
                    + "first complete a parameter search."
                )

            # project onto the search path submodel with `terms` number of terms
            return self.path[terms]

        # raise error
        else:
            raise UserWarning("Please pass either a list, tuple, or integer.")

    def project_names(self, term_names: Sequence[str]) -> SubModel:
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using PyTorch and
        Adam for the optimisation.

        Args:
            term_names (Sequence[str]): Collection of strings containing the
            names of the parameters to include the submodel **not** including
            the intercept term

        Returns:
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # copy term names to avoid mutating the input
        term_names_ = copy.copy(term_names)

        # if projecting onto the reference model, simply return it
        if set(term_names_) == set(self.model.common_terms):
            return SubModel(
                model=self.model,
                idata=self.idata,
                loss=0,
                size=len(self.model.common_terms),
                term_names=term_names_,
            )

        # build restricted bambi model
        new_model = self._build_restricted_model(term_names=term_names_)

        # extract the design matrix from the model
        if new_model._design.common:
            X = new_model._design.common.design_matrix

            # Add offset columns to their own design matrix
            # Remove them from the common design matrix.
            if hasattr(new_model, "offset_terms"):  # pragma: no cover
                for term in new_model.offset_terms:
                    term_slice = new_model._design.common.slices[term]
                    X = np.delete(X, term_slice, axis=1)

        # build new term_names (add dispersion parameter if included)
        term_names_ = self._extend_term_names(
            new_model=new_model, term_names=term_names_
        )

        # compute projected posterior
        projected_posterior, loss = self.solver.solve(term_names=term_names_, X=X)

        # add observed data component of projected idata
        observed_data = {
            self.response_name: self.idata.observed_data.get(self.response_name)
            .to_dict()
            .get("data")
        }

        # build idata object for the projected model
        new_idata = az.data.from_dict(
            posterior=projected_posterior,
            observed_data=observed_data,
        )

        # compute the log-likelihood of the new submodel and add to idata
        log_likelihood = self.compute_model_log_likelihood(
            model=new_model, idata=new_idata
        )
        new_idata.add_groups(
            log_likelihood={self.response_name: log_likelihood},
            dims={self.response_name: [f"{self.response_name}_dim_0"]},
        )

        # build SubModel object and return
        sub_model = SubModel(
            model=new_model,
            idata=new_idata,
            loss=loss,
            size=len(new_model.common_terms),
            term_names=term_names,
        )
        return sub_model

    def compute_model_log_likelihood(self, model, idata):
        # extract observed data
        response_dims = model.response.name + "_obs"
        obs_array = xr.DataArray(model.data[model.response.name], dims=response_dims)

        # make insample latent predictions
        preds = model.predict(idata, kind="mean", inplace=False).posterior[
            f"{model.response.name}_mean"
        ]
        linear_preds = model.family.link.link(preds)

        if model.family.name == "gaussian":
            # initialise probability distribution object
            dist = XrContinuousRV(
                stats.norm, linear_preds, idata.posterior[f"{model.response.name}_sigma"]
            )

        # compute log likelihood of model
        log_likelihood = dist.logpdf(obs_array).transpose(*("chain", "draw", ...))
        return log_likelihood

    def _build_restricted_formula(self, term_names: List[str]) -> str:
        """Build the formula for the restricted model."""

        formula = (
            f"{self.response_name} ~ " + " + ".join(term_names)
            if len(term_names) > 0
            else f"{self.response_name} ~ 1"
        )
        return formula

    def _build_restricted_model(self, term_names: List[str]) -> bmb.Model:
        """Build the restricted model in Bambi."""

        new_formula = self._build_restricted_formula(term_names=term_names)
        new_model = bmb.Model(new_formula, self.model.data, family=self.ref_family)
        return new_model

    def _extend_term_names(
        self, new_model: bmb.Model, term_names: List[str]
    ) -> List[str]:
        """Extend the model term names to include dispersion terms."""

        # add intercept term if present
        if new_model.intercept_term:
            term_names.insert(0, "Intercept")

        # add the auxiliary parameters
        if self.priors:
            aux_params = [f"{self.response_name}_{str(k)}" for k in self.priors]
            term_names += aux_params
        return term_names
