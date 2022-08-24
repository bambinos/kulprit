"""Optimisation module."""

from typing import Optional

import numpy as np

from kulprit.data.submodel import SubModel
from bambi.models import Model
from arviz import InferenceData

import pymc as pm
import bambi as bmb

import pandas as pd


class Solver:
    """The primary solver class, used to perform the projection."""

    def __init__(self, model: Model, idata: InferenceData) -> None:
        """Initialise the main solver object."""

        # log the reference model and inference data objects
        self.ref_model = model
        self.pymc_model = self.ref_model.backend.model
        self.ref_idata = idata

        # log the reference model's response name and family
        self.response_name = self.ref_model.response.name
        self.ref_family = self.ref_model.family.name

    @property
    def new_data(self) -> pd.DataFrame:
        """Build new dataset for projection.

        This new dataset is identical to the reference model dataset in the
        covariates, but the originally observed variate has been replaced by the
        point estimate predictions from the reference model.
        """

        # make in-sample predictions with the reference model if not available
        if "posterior_predictive" not in self.ref_idata.groups():
            self.ref_model.predict(self.ref_idata, kind="pps", inplace=True)

        # extract insample predictions
        preds = self.ref_idata.posterior_predictive[self.response_name].mean(
            ["chain", "draw"]
        )

        # build new dataframe
        new_data = self.ref_model.data.copy()
        new_data[self.response_name] = preds
        return new_data

    def _build_restricted_formula(self, term_names: list) -> str:
        """Build the formula for the restricted model."""

        formula = (
            f"{self.response_name} ~ " + " + ".join(term_names)
            if len(term_names) > 0
            else f"{self.response_name} ~ 1"
        )
        return formula

    def _build_restricted_model(self, term_names: list) -> Model:
        """Build the restricted model in Bambi."""

        new_formula = self._build_restricted_formula(term_names=term_names)
        new_model = bmb.Model(new_formula, self.new_data, family=self.ref_family)
        new_model.build()
        return new_model

    def _infmean(self, input_array: np.ndarray) -> float:
        """Return the mean of the finite values of the array.

        This method is taken from pymc.variational.Inference.
        """

        input_array = input_array[np.isfinite(input_array)].astype("float64")
        if len(input_array) == 0:
            return np.nan
        else:
            return np.mean(input_array)

    def solve(
        self,
        term_names: list,
        num_steps: Optional[int] = 5_000,
        obj_n_mc: Optional[float] = 10,
    ) -> SubModel:
        """The primary projection method in the procedure.

        The projection is performed with a mean-field approximation to variational
        inference rather than concatenating posterior draw-wise optimisation
        solutions as is suggested by Piironen (2018).
        """

        # if projecting onto the reference model, simply return it
        if set(term_names) == set(self.ref_model.common_terms.keys()):
            return SubModel(
                model=self.ref_model,
                idata=self.ref_idata,
                elbo=np.inf,
                size=len(self.ref_model.common_terms),
                term_names=term_names,
            )

        # build restricted model
        new_model = self._build_restricted_model(term_names=term_names)
        new_pymc_model = new_model.backend.model

        # perform mean-field MLE
        with new_pymc_model:
            approx = pm.MeanField()
            inference = pm.KLqp(approx, beta=0.0)
            mean_field = inference.fit(n=num_steps, obj_n_mc=obj_n_mc, progressbar=False)

        # compute the LOO-CV predictive performance of the submodel
        num_draws = (
            self.ref_idata.posterior.dims["chain"]
            * self.ref_idata.posterior.dims["draw"]
        )
        trace = mean_field.sample(num_draws, return_inferencedata=False)

        # first obtain the aesara observed RVs
        new_obs_rvs = new_pymc_model.named_vars[self.response_name]
        old_obs_rvs = self.pymc_model.named_vars[self.response_name]
        # and then we replace the observations in the new model for those in the reference model
        new_pymc_model.rvs_to_values[new_obs_rvs] = self.pymc_model.rvs_to_values[
            old_obs_rvs
        ]
        new_idata = pm.to_inference_data(
            trace=trace, model=new_pymc_model, log_likelihood=True
        )
        coords_to_drop = [
            dim for dim in new_idata.posterior.dims if dim.endswith("_dim_0")
        ]
        new_idata.posterior = new_idata.posterior.squeeze(coords_to_drop).reset_coords(
            coords_to_drop, drop=True
        )
        # compute the average elbo
        elbo = self._infmean(mean_field.hist[max(0, num_steps - 1000) : num_steps + 1])

        # build SubModel object and return
        sub_model = SubModel(
            model=new_model,
            idata=new_idata,
            elbo=elbo,
            size=len([term for term in term_names if term != "Intercept"]),
            term_names=term_names,
        )
        return sub_model
