"""Base projection class."""

import dataclasses

import arviz as az
import numpy as np
import torch

from .data import ModelData
from .data.utils import _posterior_to_points
from .families import Family
from .formatting import spacify, multilinify
from .projection import _DivLoss, _KulOpt
from .utils import _extract_insample_predictions, _compute_elpd, _compute_log_likelihood


class Projector:
    def __init__(self, model, inferencedata=None):
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Args:
            model (bambi.models.Model): The referemce GLM model to project
            inferencedata (arviz.InferenceData): The arViz InferenceData object
                of the fitted reference model
        """

        # build posterior if unavailable
        if inferencedata is None:
            inferencedata = model.fit()
        # log the underlying backend model
        backend = model.backend
        # instantiate family object from model
        family = Family.create(model)
        # define the link function object for the reference model
        link = model.family.link
        # extract covariate and variate names
        term_names = list(model.term_names)
        common_terms = list(model.common_terms.keys())
        response_name = model.response.name
        # extract data from the fitted bambi model
        predictions = model.predict(
            idata=inferencedata, inplace=False, kind="pps"
        ).posterior_predictive[response_name]
        X = torch.from_numpy(model._design.common.design_matrix).float()
        y = torch.from_numpy(model._design.response.design_vector).float()
        design = model._design
        has_intercept = model.intercept_term is not None
        if not has_intercept:
            raise NotImplementedError(
                "The procedure currently only supports reference models with "
                + "an intercept term."
            )
        # extract some key dimensions needed for optimisation
        num_obs, num_terms = model._design.common.design_matrix.shape
        model_size = len(common_terms)  # note that model size ignores intercept
        num_draws = (
            inferencedata.posterior.dims["chain"] * inferencedata.posterior.dims["draw"]
        )  # to do: test this for edge cases
        # set the reference model's distance to itself as zero and compute ELPD
        dist_to_ref_model = torch.tensor(0.0)
        elpd = az.loo(inferencedata)

        # build full model object
        self.ref_model = ModelData(
            X=X,
            y=y,
            backend=backend,
            design=design,
            link=link,
            family=family,
            term_names=term_names,
            common_terms=common_terms,
            response_name=response_name,
            num_obs=num_obs,
            num_terms=num_terms,
            num_draws=num_draws,
            model_size=model_size,
            has_intercept=has_intercept,
            dist_to_ref_model=dist_to_ref_model,
            elpd=elpd,
            inferencedata=inferencedata,
            predictions=predictions,
        )

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        msg = (
            f"Projector with reference model of {self.ref_model.num_terms} terms.\n"
            f"Terms:{spacify(multilinify(self.ref_model.term_names, ''))}\n\n"
        )
        return msg

    def __getitem__(self, model_size):  # pragma: no cover
        """Extract the submodel with given `model_size`."""

        raise NotImplementedError

    def _build_restricted_model(self, model_size=None):
        """Build a restricted model from a reference model given some model size

        Args:
            model_size (int): The number of parameters to use in the restricted model

        Returns:
            kulprit.ModelData: A restricted model with `model_size` terms
        """

        if model_size == self.ref_model.model_size or model_size is None:
            # if `model_size` is same as the reference model, simply copy the ref_model
            return dataclasses.replace(
                self.ref_model, inferencedata=None, predictions=None
            )

        # test model_size in case of misuse
        if model_size < 0 or model_size > self.ref_model.model_size:
            raise UserWarning(
                "`model_size` parameter must be non-negative and less than size of"
                + f" the reference model, instead received {model_size}."
            )

        # get the variable names of the best model with `model_size` parameters
        restricted_common_terms = self.ref_model.common_terms[:model_size]
        if model_size > 0:  # pragma: no cover
            # extract the submatrix from the reference model's design matrix
            X_res = torch.from_numpy(
                np.column_stack(
                    [
                        self.ref_model.design.common[term]
                        for term in restricted_common_terms
                    ]
                )
            ).float()
            # manually add intercept to new design matrix
            X_res = torch.hstack((torch.ones(self.ref_model.num_obs, 1), X_res))
        else:
            # intercept-only model
            X_res = torch.ones(self.ref_model.num_obs, 1).float()

        # update common term names and dimensions and build new ModelData object
        _, num_terms = X_res.shape
        restricted_term_names = ["Intercept"] + restricted_common_terms
        res_model = dataclasses.replace(
            self.ref_model,
            X=X_res,
            num_terms=num_terms,
            model_size=model_size,
            term_names=restricted_term_names,
            common_terms=restricted_common_terms,
            inferencedata=None,
            predictions=None,
        )
        # ensure correct dimensions
        assert res_model.X.shape == (self.ref_model.num_obs, model_size + 1)
        return res_model

    def _build_idata(self, theta_perp, model, disp_perp=None):
        """Convert some set of pytorch tensors into an ArviZ InferenceData object.

        Args:
            theta_perp (torch.tensor): Restricted parameter posterior projections,
                including the intercept term
            model (kulprit.ModelData): The restricted ModelData object whose
                posterior to build
            disp_perp (torch.tensor): Restricted model dispersions parameter
                posterior projections

        Returns:
            arviz.InferenceData: Restricted model posterior
        """

        # build posterior dictionary from projected parameters
        posterior = {
            f"{term}": theta_perp[:, i] for i, term in enumerate(model.term_names)
        }
        if disp_perp is not None:
            disp_dict = {
                f"{model.response_name}_sigma": disp_perp,
                f"{model.response_name}_sigma_log__": torch.log(disp_perp),
            }
            posterior.update(disp_dict)

        # build points data from the posterior dictionaries
        points = _posterior_to_points(posterior, self.ref_model)
        # compute log-likelihood of projected model from this posterior
        log_likelihood = _compute_log_likelihood(model.backend, points)

        idata = az.data.from_dict(posterior=posterior, log_likelihood=log_likelihood)
        return idata

    def project(
        self,
        model_size=None,
        num_iters=200,
        learning_rate=0.01,
    ):
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using PyTorch and
        Adam for the optimisation.

        Example:
            When ``num_vars = 0``, the reference model is projected onto the
            model with only the intercept term and no covariates.

        Args:
            num_vars (int): The number parameters to use in the restricted
                model, **not** including the intercept term, must be greater
                than or equal to zero and less than or equal to the number of
                parameters in the reference model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate

        Returns:
            torch.tensor: Restricted projection of the reference parameters
        """

        # test `model_size` input
        if model_size is None:
            model_size = self.ref_model.model_size
        elif model_size < 0:
            raise UserWarning(
                "`model_size` parameter must be non-negative, received value "
                + f"{model_size}."
            )
        elif model_size > self.ref_model.model_size:
            raise UserWarning(
                "`model_size` parameter cannot be greater than the size of the"
                + f" reference model ({self.ref_model.model_size}), received"
                + f" value {model_size}."
            )
        # build restricted model object
        res_model = self._build_restricted_model(model_size)
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

        # extract projected parameters and final KL divergence from the solver
        theta_perp = list(solver.parameters())[0].data
        res_model.dist_to_ref_model = loss.item()
        # if the reference family has dispersion parameters, project them
        if self.ref_model.family.has_disp_params:
            # build posterior with just the covariates
            res_model.inferencedata = self._build_idata(theta_perp, res_model)
            # project dispersion parameters
            disp_perp = self.ref_model.family._project_disp_params(
                self.ref_model, res_model
            )
        # build the complete restricted model posterior
        res_model.inferencedata = self._build_idata(theta_perp, res_model, disp_perp)
        # ELPD of restricted model
        res_model.elpd = az.loo(res_model.inferencedata)
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
