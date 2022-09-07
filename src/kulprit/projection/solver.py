"""Optimisation module."""

from typing import List, Optional

from kulprit.data.submodel import SubModel
from kulprit.projection.likelihood import LIKELIHOODS

import arviz as az
import bambi as bmb

import numpy as np

from scipy.optimize import minimize


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

        # define sampling options
        self.num_chain = self.ref_idata.posterior.dims["chain"]
        self.num_samples = self.num_chain * 100

        try:
            # define the negative log likelihood function of the submodel
            self.neg_log_likelihood = LIKELIHOODS[self.ref_family]
        except KeyError:
            raise NotImplementedError from None

    @property
    def pps(self):
        # make in-sample predictions with the reference model if not available
        if "posterior_predictive" not in self.ref_idata.groups():
            self.ref_model.predict(self.ref_idata, kind="pps", inplace=True)

        pps = az.extract_dataset(
            self.ref_idata,
            group="posterior_predictive",
            var_names=[self.response_name],
            num_samples=self.num_samples,
        )[self.response_name].values.T
        return pps

    def linear_predict(
        self,
        beta_x: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the latent predictor of the submodel.

        Args:
            beta_x (np.ndarray): The model's projected posterior
            X (np.ndarray): The model's common design matrix
            x_offset (np.ndarray): Offset terms in the model is included

        Return
            np.ndarray: Point estimate of the latent predictor using the single draw
                from the posterior and the model's design matrix
        """

        linear_predictor = np.zeros(shape=(X.shape[0],))

        # Contribution due to common terms
        if X is not None:

            if len(beta_x.shape) > 1:  # pragma: no cover
                raise NotImplementedError(
                    "Currently this method only works for single samples."
                )

            # 'contribution' is of shape:
            # * (obs_n, ) for univariate
            contribution = np.dot(X, beta_x.T).T
            linear_predictor += contribution

        # return the latent predictor
        return linear_predictor

    def _init_optimisation(self, term_names: List[str]) -> List[float]:
        """Initialise the optimisation with the reference posterior means."""
        init = np.hstack(
            [
                self.ref_idata.posterior.mean(["chain", "draw"])[term].values
                for term in term_names
            ]
        )
        return init

    def _build_bounds(self, init: List[float]) -> list:
        """Build bounds for the parameters in the optimimsation.

        This method is used to ensure that dispersion or other auxiliary
        parameters present in certain families remain within their valid regions.

        Args:
            init (List[float]): The list of initial parameter values

        Returns
            List[Tuple(float)]: The upper and lower bounds for each initialised
                parameter in the optimisation
        """

        # build bounds based on family
        if self.ref_family in ["gaussian"]:
            # account for the dispersion parameter
            bounds = [(None, None)] * (init.size - 1) + [(0, None)]
        return bounds

    def objective(
        self,
        params: np.ndarray,
        obs: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Variational projection predictive objective function.

        This is negative log-likelihood of the restricted model but evaluated
        on samples of the posterior predictive distribution of the reference model.
        Formally, this objective function implements Equation 1 of mean-field
        projection predictive inference as defined [here](https://www.hackmd.io/
        @yannmcl/H1CZPjE1i).

        Args:
            params (list): The optimisation parameters mean values
            obs (list): One sample from the posterior predictive distribution of
                the reference model

        Returns:
            float: The negative log-likelihood of the reference posterior
                predictive under the restricted model
        """

        # Gaussian observation likelihood
        if self.ref_family == "gaussian":
            linear_predictor = self.linear_predict(beta_x=params[:-1], X=X)
            neg_llk = self.neg_log_likelihood(
                points=obs, mean=linear_predictor, sigma=params[-1]
            )
        # Binomial observation likelihood
        elif self.ref_family == "binomial":
            trials = self.model.response.data[:, 1]
            linear_predictor = self.linear_predict(beta_x=params, X=X)
            mean = self.model.link.linkinv(linear_predictor)
            neg_llk = self.neg_log_likelihood(points=obs, mean=mean, trials=trials)
        # Poisson observation likelihood
        elif self.ref_family == "poisson":
            linear_predictor = self.linear_predict(beta_x=params, X=X)
            mean = self.model.link.linkinv(linear_predictor)
            neg_llk = self.neg_log_likelihood(points=obs, mean=mean)
        return neg_llk

    def solve(self, term_names: List[str], X: np.ndarray) -> SubModel:
        """The primary projection method in the procedure.

        The projection is performed with a mean-field approximation rather than
        concatenating posterior draw-wise optimisation solutions as is suggested
        by Piironen (2018). For more information, kindly read [this tutorial](h
        ttps://www.hackmd.io/@yannmcl/H1CZPjE1i).

        Args:
            term_names (List[str]): The names of the terms to project onto in
                the submodel
            X (np.ndarray): The common term design matrix of the submodel

        Returns:
            SubModel: The projected submodel object
        """

        # initialise the optimisation
        init = self._init_optimisation(term_names=term_names)

        # build the optimisation parameter bounds
        bounds = self._build_bounds(init)

        # perform mean-field variational projection predictive inference
        res_posterior = []
        objectives = []
        for obs in self.pps:
            opt = minimize(
                self.objective,
                args=(obs, X),
                x0=init,  # use reference model posterior as initial guess
                bounds=bounds,  # apply bounds
                method="powell",
            )
            res_posterior.append(opt.x)
            objectives.append(opt.fun)

        # compile the projected posterior
        res_samples = np.vstack(res_posterior).T
        posterior = {term: samples for term, samples in zip(term_names, res_samples)}
        posterior.update(
            (key, value.reshape(self.num_chain, 100)) for key, value in posterior.items()
        )

        # compute the average loss
        loss = np.mean(objectives)

        return posterior, loss
