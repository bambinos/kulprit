"""Bambi model structure data classes."""

import torch

from bambi.models import Model


class ModelStructure:
    def __init__(self, model: Model) -> None:
        """Initiate ModelStructure object from Bambi model.

        Args:
            model (bambi.models.Model): A Bambi model whose structure to define

        Attributes:
            X (torch.tensor): Model design matrix
            y (torch.tensor): Model variate observations
            backend (pymc.Model): The PyMC model backend
            design (formulae.matrices.DesignMatrices): The formulae design matrix
                object underpinning the GLM
            link (bambi.families.Link): GLM link function object
            term_names (list): List of model covariates in their order of
                appearance **not** including the `Intercept` term
            common_terms (list): List of all terms in the model in order of
                appearance (includes the `Intercept` term)
            response_name (str): The name of the response given to the Bambi model
            num_obs (int): Number of data observations
            num_terms (int): Number of variables observed, and equivalently the
                number of common terms in the model (including intercept)
            num_draws (int): Number of posterior draws in the model
            model_size (int): Number of common terms in the model (terms not
                including the intercept)
            has_intercept (bool): Flag whether intercept included in model
            predictions (arviz.InferenceData): In-sample model predictions
        """

        # build structure
        self.structure_factory(model)

    def structure_factory(self, model: Model) -> None:
        """Build structure variables for a given Bambi model.

        Args:
            model (bambi.models.Model): A Bambi model whose structure to define
        """

        # store the prediction function
        self.predict = model.predict

        # log the underlying backend model
        self.backend = model.backend

        # define model architecture
        self.architecture = "glm" if len(model.group_specific_terms) == 0 else "glmm"

        # define the link function and family of the reference model
        self.link = model.family.link
        self.family = model.family.name

        # extract covariate and variate names
        self.term_names = list(model.term_names)
        self.common_terms = list(model.common_terms.keys())
        self.response_name = model.response.name
        self.design = model._design
        self.has_intercept = model.intercept_term is not None

        # extract data from the fitted bambi model
        self.X = torch.from_numpy(model._design.common.design_matrix).float()
        self.y = torch.from_numpy(model._design.response.design_matrix).float()
        self.predictions = None

        # extract some key dimensions needed for optimisation
        self.num_obs, self.num_terms = model._design.common.design_matrix.shape
        self.model_size = len(self.common_terms)  # does not include intercept
