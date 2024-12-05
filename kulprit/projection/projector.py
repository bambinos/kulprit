# pylint: disable=too-many-instance-attributes
"""Base projection class."""

import collections
from typing import Optional, Sequence, Union

import arviz as az
import bambi as bmb
import numpy as np

from kulprit.data.submodel import SubModel
from kulprit.projection.pymc_io import (
    compile_mllk,
    compute_llk,
    compute_new_model,
    get_model_information,
)
from kulprit.projection.solver import solve


class Projector:
    def __init__(
        self,
        model: bmb.Model,
        idata: az.InferenceData,
        num_samples: int,
        has_intercept: bool,
        noncentered: bool,
        path: Optional[dict] = None,
    ) -> None:
        """Reference model builder for projection predictive model selection.

        This class handles the core projection methods of the model selection procedure.
        Note that throughout the procedure, variables with names of the form ``*_ast`` belong to
        the reference model while variables with names like ``*_perp`` belong to the restricted
        model. This is to preserve notation choices from previous papers on the topic.

        Parameters:
        ----------
        model : bambi model
            The reference model to be projected.
        idata : InferenceData
            The inference data object corresponding to the reference model.
        path : dict
            An optional search path dictionary, initialized to None and assigned by the
        ProjectionPredictive parent object following a search for efficient submodel retrieval.
        """

        # log reference model and reference inference data object
        self.model = model
        self.idata = idata
        self.has_intercept = has_intercept
        self.num_samples = num_samples
        self.noncentered = noncentered

        # log properties of the reference Bambi model
        self.response_name = model.response_component.term.name
        self.ref_family = self.model.family.name
        self.priors = self.model.constant_components

        self.observed_data = self.get_observed_data()
        self.pps = self.get_pps()
        self.base_terms = self.get_base_terms()

        # log properties of the reference PyMC model
        self.pymc_model = model.backend.model
        self.all_terms = [fvar.name for fvar in self.pymc_model.free_RVs]
        self.ref_var_info = get_model_information(self.pymc_model)

        self.prev_models = []
        # log search path
        self.path = path

    def project(
        self,
        terms: Union[Sequence[str], int],
    ) -> SubModel:
        """Wrapper function for projection method.

        Parameters:
        ----------
        terms : (Union[Sequence[str], int])
            Collection of strings containing the names of the parameters to include the submodel,
        or the number of parameters to include in the submodel, not including the intercept term

        Returns:
        -------
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # project terms by name
        if isinstance(terms, collections.abc.Sequence):
            # if not a list, cast to list
            if not isinstance(terms, list):
                terms = list(terms)

            # test `terms` input
            ref_terms = list(
                self.model.components[self.model.family.likelihood.parent].terms.keys()
            )

            if not set(terms).issubset(set(ref_terms)):
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
        """Primary projection method for reference model.

        The projection is defined as the values of the submodel parameters minimizing the
        Kullback-Leibler divergence between the submodel and the reference model.
        This is achieved by maximizing the log-likelihood of the submodel wrt the predictions of
        the reference model.

        Parameters:
        ----------
        term_names : Sequence[str]
            Collection of strings containing the names of the parameters to include the submodel,
            not including the intercept term

        Returns:
        -------
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        term_names_ = self.base_terms + term_names
        new_model = compute_new_model(
            self.pymc_model, self.noncentered, self.ref_var_info, self.all_terms, term_names_
        )
        print("free_rvs", new_model.free_RVs)
        model_log_likelihood, old_y_value, obs_rvs = compile_mllk(new_model)
        initial_guess = np.concatenate(
            [np.ravel(value) for value in new_model.initial_point().values()]
        )
        var_info = get_model_information(new_model)

        new_idata, loss = solve(
            model_log_likelihood,
            self.pps,
            initial_guess,
            var_info,
        )
        # restore obs_rvs value in the model
        new_model.rvs_to_values[obs_rvs] = old_y_value

        # Add observed data to the projected InferenceData object
        new_idata.add_groups(observed_data=self.observed_data)
        # Add log-likelihood to the projected InferenceData object
        new_idata.add_groups(log_likelihood=compute_llk(new_idata, new_model))

        # build SubModel object and return

        sub_model = SubModel(
            model=new_model,
            idata=new_idata,
            loss=loss,
            size=len(new_model.free_RVs) - len(self.base_terms),
            term_names=term_names,
            has_intercept=self.has_intercept,
        )
        return sub_model

    def get_observed_data(self):
        """Extract the observed data from the reference model."""
        observed_data = {
            self.response_name: self.idata.observed_data.get(self.response_name)
            .to_dict()
            .get("data")
        }
        return az.convert_to_dataset(observed_data)

    def get_pps(self):
        """Extract the posterior predictive samples from the reference model."""
        if "posterior_predictive" not in self.idata.groups():
            self.model.predict(self.idata, kind="response", inplace=True)

        pps = az.extract(
            self.idata,
            group="posterior_predictive",
            var_names=[self.response_name],
            num_samples=self.num_samples,
        ).values.T
        return pps

    def get_base_terms(self):
        """Extend the model term names to include dispersion terms."""

        base_terms = []
        # add intercept term if present
        if self.has_intercept:
            base_terms.append("Intercept")

        # add the auxiliary parameters
        if self.priors:
            aux_params = [f"{str(k)}" for k in self.priors]
            base_terms += aux_params
        return base_terms
