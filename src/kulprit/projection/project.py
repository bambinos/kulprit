"""Projection class."""

import arviz as az
import torch

from .optimise import _DivLoss, _KulOpt
from ..data import ModelData
from ..utils import (
    _build_restricted_model,
    _extract_theta_perp,
    _extract_insample_predictions,
)
from ..families import Family


class Projector:
    def __init__(self, model, posterior):
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Args:
            model (bambi.models.Model): The Bambi GLM model of interest
            posterior (arviz.InferenceData): The posterior arViz object of the
                fitting Bambi model
        """

        # define key model attributes
        family = Family.create(model.family.name)
        link = model.family.link
        response_name = model.response.name
        predictions = model.predict(idata=posterior, inplace=False, kind="pps").posterior_predictive[response_name]
        X = torch.from_numpy(model._design.common.design_matrix).float()
        y = torch.from_numpy(model._design.response.design_vector).float()
        data = model.data
        cov_names = [cov for cov in model.term_names if cov in model.data.columns]
        n, m = model._design.common.design_matrix.shape
        s = posterior.posterior.dims["chain"] * posterior.posterior.dims["draw"]
        has_intercept = model.intercept_term is not None
        # build full model object
        self.full_model = ModelData(
            X=X,
            y=y,
            data=data,
            link=link,
            family=family,
            cov_names=cov_names,
            n=n,
            m=m,
            s=s,
            has_intercept=has_intercept,
            posterior=posterior,
            predictions=predictions,
        )

    def project(
        self,
        cov_names=None,
        num_iters=200,
        learning_rate=0.01,
    ):
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using a PyTorch
        neural network architecture for efficiency.

        Todo:
            * Project dispersion parameters if present in reference distribution

        Args:
            cov_names (list): The names parameters to use in the restricted
                model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate

        Returns:
            torch.tensor: Restricted projection of the reference parameters
        """

        # build restricted model object
        res_model = _build_restricted_model(self.full_model, cov_names)

        # extract restricted design matrix
        X_perp = res_model.X
        # extract reference model posterior predictions
        y_ast = _extract_insample_predictions(self.full_model)

        # build optimisation solver object
        solver = _KulOpt(res_model)
        solver.zero_grad()
        opt = torch.optim.Adam(solver.parameters(), lr=learning_rate)
        criterion = _DivLoss(res_model.family)
        # run optimisation loop
        for _ in range(num_iters):
            opt.zero_grad()
            y_perp = solver(X_perp)
            loss = criterion(y_ast, y_perp)
            loss.backward()
            opt.step()

        # extract projected parameters from the solver
        theta_perp = _extract_theta_perp(solver, res_model.cov_names)
        return theta_perp

    def plot_projection(
        self,
        cov_names=None,
        num_iters=200,
        learning_rate=0.01,
    ):
        """Plot Kullback-Leibler projection onto a parameter subset.

        Args:
            cov_names (list): The names parameters to use in the restricted
                model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate
        """

        theta_perp = self.project(cov_names)
        az.plot_posterior(theta_perp)
