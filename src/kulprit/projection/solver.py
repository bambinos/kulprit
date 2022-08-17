"""Optimisation module."""

from typing import Optional, Tuple

import numpy as np

from kulprit.data.submodel import SubModel


import pymc as pm
import bambi as bmb
import arviz as az

import pandas as pd


class Solver:
    """The primary solver class, used to perform the projection."""

    def __init__(self, model, idata):
        """Initialise the main solver object."""

        # log the reference model and inference data objects
        self.ref_model = model
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

        # extract in-sample predictions
        preds = (
            self.ref_idata.posterior_predictive.stack(samples=("chain", "draw"))
            .transpose(*("samples", ...))[self.response_name]
            .values.mean(0)
        )

        # build new dataframe
        new_data = self.ref_model.data.copy()
        new_data[self.response_name] = preds
        return new_data

    def _build_restricted_formula(self, term_names: list) -> str:
        """Build the formula for the restricted model."""

        formula = f"{self.response_name} ~ " + " + ".join(term_names)
        return formula

    def _build_restricted_model(self, term_names: list) -> bmb.Model:
        """Build the restricted model in Bambi."""

        new_formula = self._build_restricted_formula(term_names=term_names)
        new_model = bmb.Model(new_formula, self.new_data, family=self.ref_family)
        new_model.build()
        return new_model

    def _infmean(self, input_array):
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
    ) -> Tuple[bmb.Model, az.InferenceData]:
        """The primary projection method in the procedure.

        The projection is performed with a mean-field approximation to variational
        inference rather than concatenating posterior draw-wise optimisation
        solutions as is suggested by Piironen (2018).
        """

        # build restricted model
        new_model = self._build_restricted_model(term_names=term_names)
        underlying_model = new_model.backend.model

        # perform mean-field MLE
        with underlying_model:
            approx = pm.MeanField()
            inference = pm.KLqp(approx, beta=0.0)
            mean_field = inference.fit(n=num_steps, obj_n_mc=obj_n_mc, progressbar=False)

        # compute the LOO-CV predictive performance of the submodel
        num_draws = (
            self.ref_idata.posterior.dims["chain"]
            * self.ref_idata.posterior.dims["draw"]
        )
        trace = mean_field.sample(num_draws, return_inferencedata=False)
        new_idata = pm.to_inference_data(
            trace=trace, model=underlying_model, log_likelihood=True
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
