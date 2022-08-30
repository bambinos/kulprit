"""Optimisation module."""

from typing import List

from kulprit.data.submodel import SubModel

import arviz as az
import bambi as bmb

import numpy as np

from scipy.optimize import minimize
from scipy import stats


class Solver:
    """The primary solver class, used to perform the projection."""

    def __init__(
        self,
        model: bmb.Model,
        idata: az.InferenceData,
    ) -> None:
        """Initialise the main solver object."""

        # log the reference model and inference data objects
        self.ref_model = model
        self.ref_idata = idata

        # log the reference model's response name and family
        self.response_name = self.ref_model.response.name
        self.ref_family = self.ref_model.family.name
        self.priors = self.ref_model.family.likelihood.priors

        # define sampling options
        self.num_obs, _ = self.ref_model.data.shape
        self.num_chain = self.ref_idata.posterior.dims["chain"]
        self.num_draw = self.ref_idata.posterior.dims["draw"]
        self.num_samples = self.num_chain * self.num_draw

    @property
    def pps(self):
        # make in-sample predictions with the reference model if not available
        if "posterior_predictive" not in self.ref_idata.groups():
            self.ref_model.predict(self.ref_idata, kind="pps", inplace=True)

        pps = az.extract_dataset(
            self.ref_idata,
            group="posterior_predictive",
            var_names=[self.response_name],
            num_samples=100,
        )[self.response_name].values.T
        return pps

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
        new_model = bmb.Model(new_formula, self.ref_model.data, family=self.ref_family)
        return new_model

    def _init_optimisation(self, term_names: List[str]) -> List[float]:
        """Initialise the optimisation with the reference posterior means."""
        init = (
            self.ref_idata.posterior.mean(["chain", "draw"])[term_names]
            .to_array()
            .values
        )
        return init

    def _build_new_term_names(
        self, new_model: bmb.Model, term_names: List[str]
    ) -> List[str]:
        """Extend the model term names to include dispersion terms."""

        # add intercept term if present
        if new_model.intercept_term:
            term_names += ["Intercept"]

        # add the auxiliary parameters
        if self.priors:
            aux_params = [f"{self.response_name}_{str(k)}" for k in self.priors.keys()]
            term_names += aux_params
        return term_names

    def _build_bounds(self, init: List[float]) -> list:
        if self.ref_family in ["gaussian", "beta"]:
            # account for the dispersion parameter
            bounds = [(None, None)] * (init.size - 1) + [(0, None)]
        elif self.ref_family == "t":
            # account for the dispersion parameter
            bounds = [(None, None)] * (init.size - 2) + [(0, None)] * 2
        else:
            return NotImplementedError

        return bounds

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

    def compute_log_likelihood(self, model_logp, points):
        """Compute log-likelihood of some data points given a PyMC model.

        Args:
            points (list) : List of dictionaries, where each dictionary is a named
                sample of all parameters in the model

        Returns:
            dict: Dictionary of log-likelihoods at each point
        """

        raise NotImplementedError

    def _log_pdf(
        self, params: np.ndarray, obs: np.ndarray, design_matrix: np.ndarray
    ) -> np.ndarray:
        """Switch method to compute log-likelihood based on family."""

        # Gaussian observation likelihood
        if self.ref_family == "gaussian":
            mu = design_matrix @ params[:-1]
            return stats.norm.logpdf(obs, mu, params[-1])

        # Student-t observation likelihood
        elif self.ref_family == "t":
            mu = design_matrix @ params[:-2]
            return stats.t.logpdf(obs, loc=mu, df=params[-1], scale=params[-2])

        # unimplemented family error
        else:
            raise NotImplementedError

    def neg_llk(
        self, params: np.ndarray, obs: np.ndarray, design_matrix: np.ndarray
    ) -> np.ndarray:
        """Variational projection predictive objective function.

        This is negative log-likelihood of the restricted model but evaluated
        on samples of the posterior predictive distribution of the reference model.
        Formally, this objective function implements Equation 1 of mean-field
        projection predictive inference as defined [here](https://www.hackmd.io/
        @yannmcl/H1CZPjE1i).

        Args:
            params (list): The optimisation parameters mean values
            obs (list): One sample from the posterior predictive distribution `p`

        Returns:
            float: The negative log-likelihood of the reference posterior
                predictive under the restricted model
        """

        # compute log-likelihood for resricted model
        logpdf = self._log_pdf(params, obs, design_matrix)
        return -np.sum(logpdf)

    def solve(self, term_names: List[str]) -> SubModel:
        """The primary projection method in the procedure.

        The projection is performed with a mean-field approximation rather than
        concatenating posterior draw-wise optimisation solutions as is suggested
        by Piironen (2018). For more information, kindly read [this tutorial](h
        ttps://www.hackmd.io/@yannmcl/H1CZPjE1i).

        Args:
            term_names (List[str]): The names of the terms to project onto in
                the submodel

        Returns:
            SubModel: The projected submodel object
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

        # build restricted bambi model
        new_model = self._build_restricted_model(term_names=term_names)

        # build submodel design matrix
        design_matrix = new_model._design.common.design_matrix

        # build new term_names (add dispersion parameter if included)
        term_names = self._build_new_term_names(
            new_model=new_model, term_names=term_names
        )

        # initialise the optimisation
        init = self._init_optimisation(term_names=term_names)
        print(init, term_names)

        # build the optimisation parameter bounds
        bounds = self._build_bounds(init)

        # perform mean-field variational projection predictive inference
        res_posterior = []
        objectives = []
        for obs in self.pps:
            opt = minimize(
                self.neg_llk,
                args=(
                    obs,
                    design_matrix,
                ),
                x0=init,  # use reference model posterior as initial guess
                bounds=bounds,  # apply bounds
                method="powell",
            )
            res_posterior.append(opt.x)
            objectives.append(opt.fun)
        res_posterior = np.vstack(res_posterior)
        res_samples = np.array(
            [
                np.random.normal(loc, std, size=(self.num_samples,))
                for loc, std in zip(res_posterior.mean(0), res_posterior.std(0))
            ]
        )
        posterior = {term: samples for term, samples in zip(term_names, res_samples)}

        # build points data from the posterior dictionaries
        # points = self.posterior_to_points(posterior)

        # # compute log-likelihood of projected model from this posterior
        # log_likelihood = self.compute_log_likelihood(self.data.structure.backend, points)

        # # reshape the log-likelihood and posterior to match reference model
        # posterior.update(
        #     (key, value.reshape(self.num_chain, self.num_draw, self.num_obs))
        #     for key, value in posterior.items()
        # )
        # log_likelihood.update(
        #     (key, value.reshape(self.num_chain, self.num_draw, self.num_obs))
        #     for key, value in log_likelihood.items()
        # )

        # add observed data component of projected idata
        observed_data = {
            self.response_name: self.ref_idata.observed_data.get(self.response_name)
            .to_dict()
            .get("data")
        }

        # build idata object for the projected model
        new_idata = az.data.from_dict(
            posterior=posterior,
            # log_likelihood=log_likelihood,
            observed_data=observed_data,
        )

        # compute the average loss
        loss = np.mean(objectives)

        # build SubModel object and return
        sub_model = SubModel(
            model=new_model,
            idata=new_idata,
            elbo=loss,
            size=len(new_model.common_terms),
            term_names=term_names,
        )
        return sub_model
