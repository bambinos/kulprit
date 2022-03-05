import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd

import arviz as az


class Projector:
    def __init__(self, model, posterior):
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        To do:
            * Fetch appropriate `kl_div` method from the `divergences` module
                according to `model.family`

        Args:
            model (bambi.models.Model): The Bambi GLM model of interest
            posterior (arviz.InferenceData): The posterior arViz object of the
                fitting Bambi model
        """
        self.model = model
        self.posterior = posterior

    def project(self, num_params=1):
        """Primary projection method for a Bayesian linear regression.

        The projection function for this method is taken from Equation (6) in
        @Piirnonen2016, and the projection of the linear model variance is from
        Equation (7).

        To do:
            * Fix `vmap` axes bug for dispersion parameter projection

        Args:
            num_params (int): The size of the restricted parameter space

        Returns:
            ndarray: Restricted rojection of the parameter parameters
        """

        def _proj_theta(theta):
            """Analytic projection of the full parameters.

            Args:
                theta (ndarray): The learned full parameters

            Returns:
                ndarray: The restricted projections of the parameters
            """

            f = X @ theta
            theta_perp = jnp.linalg.inv(X_perp.T @ X_perp) @ X_perp.T @ f
            return theta_perp

        def _proj_sigma(theta, theta_perp):
            """Analytic projection of the full noise parameter.

            Args:
                theta (ndarray): The full parameter space
                theta_perp (ndarray): The restricted parameter space

            Returns:
                ndarray: The restricted projections of the model noise
            """

            f = X @ theta
            f_perp = X_perp @ theta_perp
            sigma_perp = jnp.sqrt(sigma ** 2 + 1 / n * (f - f_perp).T @ (f - f_perp))
            return sigma_perp

        if model.family != "gaussian":
            raise UserWarning("Only Gaussian-distributed variates handled currently.")
        # extract the number of data observations
        n = X.shape[0]
        # build restricted data space
        X_perp = X[:, :num_params]
        # perform projections
        theta_perp = jax.vmap(_proj_theta, in_axes=0, out_axes=0)(theta)
        sigma_perp = jax.vmap(_proj_sigma, in_axes=(0, 0), out_axes=0)(
            theta, theta_perp
        ).mean(
            axis=0
        )  #  TODO: fix vmap for sigma projection
        return theta_perp, sigma_perp
