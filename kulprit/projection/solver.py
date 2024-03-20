"""optimization module."""

# pylint: disable=protected-access
from typing import List, Optional
import warnings

import arviz as az
import bambi as bmb
import numpy as np
import xarray as xr
import preliz as pz

from scipy.optimize import minimize

from kulprit.data.submodel import SubModel


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
        self.response_name = self.ref_model.response_name
        self.ref_family = self.ref_model.family.name

        # define sampling options
        self.num_chain = self.ref_idata.posterior.dims["chain"]
        self.num_samples = self.num_chain * 100

        if self.ref_family not in ["gaussian", "poisson", "bernoulli", "binomial"]:
            raise NotImplementedError(f"Family {self.ref_family} not supported")

    @property
    def pps(self):
        # make in-sample predictions with the reference model if not available
        if "posterior_predictive" not in self.ref_idata.groups():
            self.ref_model.predict(self.ref_idata, kind="pps", inplace=True)

        pps = az.extract(
            self.ref_idata,
            group="posterior_predictive",
            var_names=[self.response_name],
            num_samples=self.num_samples,
        ).values.T
        return pps

    def _build_bounds(self, init: List[float]) -> list:
        """Build bounds for the parameters in the optimization.

        This method is used to ensure that dispersion or other auxiliary parameters present
        in certain families remain within their valid regions.

        Parameters:
        ----------
        init : List[float]
            The list of initial parameter values

        Returns:
        -------
        List : [Tuple(float)]
            The upper and lower bounds for each initialized parameter in the optimization
        """
        eps = np.finfo(np.float64).eps

        # build bounds based on family, we are assuming that the last parameter is the dispersion
        # and that the other parameters are unbounded (like when using Gaussian priors)
        if self.ref_family in ["gaussian"]:
            # account for the dispersion parameter
            bounds = [(None, None)] * (init.size - 1) + [(eps, None)]
        elif self.ref_family in ["binomial", "poisson", "bernoulli"]:
            # no dispersion parameter, so no bounds
            bounds = [(None, None)] * (init.size)
        return bounds

    def objective(
        self,
        params: np.ndarray,
        obs: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Variational projection predictive objective function.

        This is negative log-likelihood of the restricted model but evaluated on samples of the
        posterior predictive distribution of the reference model.
        Formally, this objective function implements Equation 1 of mean-field projection predictive
        inference as defined [here](https://www.hackmd.io/@yannmcl/H1CZPjE1i).

        Parameters:
        ----------
        params : array_like
            The optimization parameters mean values
        obs : array_like
            One sample from the posterior predictive distribution of the reference model
        X : array_like
            The common term design matrix of the submodel

        Returns:
        -------
            float: The negative log-likelihood of the reference posterior predictive under the
        restricted model
        """

        # Gaussian observation likelihood
        if self.ref_family == "gaussian":
            linear_predictor = _linear_predict(beta_x=params[:-1], X=X)
            neg_llk = pz.Normal(mu=linear_predictor, sigma=params[-1])._neg_logpdf(obs)

        # Binomial observation likelihood
        elif self.ref_family == "binomial":
            trials = self.ref_model.response.data[:, 1]
            linear_predictor = _linear_predict(beta_x=params, X=X)
            probs = self.ref_model.family.link["p"].linkinv(linear_predictor)
            neg_llk = pz.Binomial(n=trials, p=probs)._neg_logpdf(obs)

        # Bernoulli observation likelihood
        elif self.ref_family == "bernoulli":
            linear_predictor = _linear_predict(beta_x=params, X=X)
            probs = self.ref_model.family.link["p"].linkinv(linear_predictor)
            neg_llk = pz.Binomial(p=probs)._neg_logpdf(obs)

        # Poisson observation likelihood
        elif self.ref_family == "poisson":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="overflow encountered in exp")
                linear_predictor = _linear_predict(beta_x=params, X=X)
                lam = self.ref_model.family.link["mu"].linkinv(np.float64(linear_predictor))
                neg_llk = pz.Poisson(mu=lam)._neg_logpdf(obs)

        return neg_llk

    def solve(self, term_names: List[str], X: np.ndarray, slices: dict) -> SubModel:
        """The primary projection method in the procedure.

        The projection is performed with a mean-field approximation rather than concatenating
        posterior draw-wise optimization solutions as is suggested by Piironen (2018).
        For more information, kindly read https://www.hackmd.io/@yannmcl/H1CZPjE1i.

        Parameters:
        ----------
        term_names : List[str]
            The names of the terms to project onto in the submodel
        X : array_like
            The common term design matrix of the submodel
        slices : dict
            Slices of the common term design matrix

        Returns:
        -------
            SubModel: The projected submodel object
        """
        # build the optimization parameter bounds
        term_values = az.extract(self.ref_idata.posterior, num_samples=self.pps.shape[0])
        init = np.stack([term_values[key].values.flatten() for key in term_names]).T
        bounds = self._build_bounds(init[0])

        # perform projection predictive inference
        res_posterior = []
        objectives = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Values in x were outside bounds")
            for idx, obs in enumerate(self.pps):
                opt = minimize(
                    self.objective,
                    args=(obs, X),
                    x0=init[idx],
                    tol=0.001,
                    bounds=bounds,
                    method="SLSQP",
                )
                res_posterior.append(opt.x)
                objectives.append(opt.fun)

        # compile the projected posterior
        res_samples = np.vstack(res_posterior)
        posterior = {term: res_samples[:, slices[term]] for term in term_names}

        # NOTE: See the draw number is hard-coded. It would be better if we could take it
        # from a better source.
        chain_n = len(self.ref_idata.posterior.coords.get("chain"))
        draw_n = 100  # len(self.ref_idata.posterior.coords.get("draw"))

        # reshape inline with reference model
        for key, value in posterior.items():
            new_shape = [chain_n, draw_n]
            coords_dict = {"chain": np.arange(chain_n), "draw": np.arange(draw_n)}

            parma_coords = self.ref_idata.posterior[key].coords
            param_dims = self.ref_idata.posterior[key].dims
            extra_dims = tuple(dim for dim in param_dims if dim not in ["chain", "draw"])

            for dim in extra_dims:
                param_coord = parma_coords.get(dim)
                coords_dict[dim] = param_coord
                new_shape.append(len(param_coord))

            # NOTE I'm not sure if this is doing the right thing. We should double check it.
            value = value.reshape(new_shape)
            posterior[key] = xr.DataArray(value, coords=coords_dict)

        posterior = xr.Dataset(posterior)

        # compute the average loss
        loss = np.mean(objectives)
        return posterior, loss


def _linear_predict(
    beta_x: np.ndarray,
    X: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Predict the latent predictor of the submodel.

    Parameters:
    ----------
    beta_x : np.ndarray
        The model's projected posterior
    X : np.ndarray
        The model's common design matrix

    Returns:
    -------
    np.ndarray: Point estimate of the latent predictor using the single draw from the
    posterior and the model's design matrix
    """

    linear_predictor = np.zeros(shape=(X.shape[0],))

    # Contribution due to common terms
    if X is not None:

        if len(beta_x.shape) > 1:
            raise NotImplementedError("Currently this method only works for single samples.")

        # 'contribution' is of shape * (obs_n, ) for univariate
        contribution = np.dot(X, beta_x.T).T
        linear_predictor += contribution

    # return the latent predictor
    return linear_predictor
