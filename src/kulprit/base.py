"""Base projection class."""

import arviz as az
import torch

from .projection import _DivLoss, _KulOpt
from .data import ModelData
from .families import Family
from .utils import (
    _build_restricted_model,
    _extract_insample_predictions,
    _build_posterior,
    _compute_elpd,
)


class Projector:
    def __init__(self, model, posterior=None):
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Args:
            model (bambi.models.Model): The referemce GLM model to project
            posterior (arviz.InferenceData): The posterior arViz object of the
                fitting Bambi model
        """

        # build posterior if unavailable
        if posterior is None:
            posterior = model.fit()
        # define key model attributes
        family = Family.create(model)
        link = model.family.link
        response_name = model.response.name
        predictions = model.predict(
            idata=posterior, inplace=False, kind="pps"
        ).posterior_predictive[response_name]
        X = torch.from_numpy(model._design.common.design_matrix).float()
        y = torch.from_numpy(model._design.response.design_vector).float()
        data = model.data
        cov_names = [cov for cov in model.term_names if cov in model.data.columns]
        response_name = model.response.name
        n, m = model._design.common.design_matrix.shape
        s = posterior.posterior.dims["chain"] * posterior.posterior.dims["draw"]
        has_intercept = model.intercept_term is not None
        if not has_intercept:
            raise NotImplementedError(
                "The procedure currently only supports reference models with "
                + "an intercept term."
            )
        dist_to_ref_model = torch.tensor(0.0)
        # to do: compute ELPD of model and add to ModelData class

        # build full model object
        self.ref_model = ModelData(
            X=X,
            y=y,
            data=data,
            link=link,
            family=family,
            cov_names=cov_names,
            response_name=response_name,
            n=n,
            m=m,
            s=s,
            has_intercept=has_intercept,
            dist_to_ref_model=dist_to_ref_model,
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

        Args:
            cov_names (list): The names parameters to use in the restricted
                model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate

        Returns:
            torch.tensor: Restricted projection of the reference parameters
        """

        # build restricted model object
        res_model = _build_restricted_model(self.ref_model, cov_names)
        # extract restricted design matrix
        X_perp = res_model.X
        # extract reference model posterior predictions
        y_ast = _extract_insample_predictions(self.ref_model)

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
        theta_perp = list(solver.parameters())[0].data
        # if the reference family has dispersion parameters, project them
        if self.ref_model.family.has_disp_params:
            # build posterior with just the covariates
            res_model.posterior = _build_posterior(theta_perp, self.ref_model)
            # project dispersion parameters
            disp_perp = self.ref_model.family._project_disp_params(
                self.ref_model, res_model
            )
        # build the complete restricted model posterior
        res_model.posterior = _build_posterior(theta_perp, self.ref_model, disp_perp)

        # todo: add Rhat convergence check for projected parameters
        # todo: compute and add ELPD to res_model object
        return res_model

    def search(self, method="forward", max_terms=None):
        """Call search method through parameter space.

        Args:
            method (str): the search heuristic to employ
            max_terms (int): the maximum number of terms to search for

        Raises:
            NotImplementedError while still in development
        """

        raise NotImplementedError(
            "This method is still in development, sorry about that!"
        )
