"""Base projection class."""


from fastcore.dispatch import typedispatch
from typing import Optional, List

import arviz as az
from arviz import InferenceData
from bambi.models import Model
import numpy as np
import torch

from .loss import KullbackLeiblerLoss
from .architecture import GLMArchitecture
from .dispersion import DispersionProjectorFactory
from ..data import ModelData
from ..data.submodel import SubModelStructure, SubModelInferenceData


class Projector:
    def __init__(
        self,
        ref_model: ModelData,
        num_iters: Optional[int] = 200,
        learning_rate: Optional[float] = 0.01,
    ) -> None:
        """Reference model builder for projection predictive model selection.

        This class handles the core projection methods of the model selection
        procedure. Note that throughout the procedure, variables with names of the form
        ``*_ast`` belong to the reference model while variables with names like
        ``*_perp`` belong to the restricted model. This is to preserve notation
        choices from previous papers on the topic.

        Args:
            ref_model (kulprit.data.ModelData): Reference model dataclass object
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate
        """

        # log reference model data object
        self.ref_model = ref_model

        # build loss function from reference model
        self.loss = KullbackLeiblerLoss(self.ref_model)

        # build dispersion parameter projector class from factory methods
        self.disp_projector = DispersionProjectorFactory(self.ref_model)

        # set optimiser parameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate

    @typedispatch
    def project(self, terms: int) -> ModelData:
        """Wrapper function for projection method when number of terms passed.

        Args:
            terms (int): The number of parameters to project onto the submodel

        Return:
            kulprit.data.ModelData: Projected submodel
        """

        # test `model_size` input
        if terms < 0:
            raise UserWarning(
                "`model_size` parameter must be non-negative, received value "
                + f"{terms}."
            )
        if terms > self.ref_model.structure.model_size:
            raise UserWarning(
                "`model_size` parameter cannot be greater than the size of the"
                + f" reference model ({self.ref_model.structure.model_size}), received"
                + f" value {terms}."
            )

        # in the future we will select the "best" `args` variables according to a
        # previously run search
        raise NotImplementedError(
            "The project method currently only accepts the names of the ",
            "parameters to project as inputs",
        )

    @typedispatch
    def project(self, terms: list) -> ModelData:
        """Wrapper function for projection method when a list is passed.

        Args:
            terms (List[int]): The names of the parameters to project onto the
                submodel

        Return:
            kulprit.data.ModelData: Projected submodel
        """

        # test args type for bad lists
        assert all(
            isinstance(s, str) for s in terms
        ), "List must include variable names as strings."

        # perform projection
        return self.project_names(term_names=terms)

    def project_names(
        self,
        term_names: List[str],
    ) -> ModelData:
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using PyTorch and
        Adam for the optimisation.

        Args:
            term_names (List[str]): The names of parameters to project onto the
                submodel, **not** including the intercept term

        Returns:
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # build restricted model object
        structure_factory = SubModelStructure(self.ref_model)
        sub_model_structure = structure_factory.create(term_names)

        # extract restricted design matrix
        X_perp = sub_model_structure.X

        # extract reference model posterior predictions
        y_ast = torch.from_numpy(
            self.ref_model.structure.predictions.stack(samples=("chain", "draw"))
            .transpose(*("samples", "y_dim_0"))
            .values
        ).float()

        # define the submodel's architecture
        # note that currently, only GLMs are supported by the procedure
        self.architecture = GLMArchitecture(sub_model_structure)

        # project parameter samples and compute distance from reference model
        theta_perp, final_loss = self.optimise(X_perp, y_ast)

        # project dispersion parameters in the model, if present
        disp_perp = self.disp_projector.forward(theta_perp, X_perp)

        # build the complete restricted model posterior
        idata_factory = SubModelInferenceData(self.ref_model)
        sub_model_idata = idata_factory.create(
            sub_model_structure, theta_perp, disp_perp
        )

        # finally, combine these projected structure and idata into `ModelData`
        sub_model = ModelData(
            structure=sub_model_structure,
            idata=sub_model_idata,
            dist_to_ref_model=final_loss,
        )
        return sub_model

    def optimise(self, X_perp, y_ast):
        """Optimisation loop in projection.

        Args:
            X_perp (torch.tensor):
            y_ast (torch.tensor):

        Returns:
            Tuple[torch.tensor, torch.tensor]: A tuple of the projected
                parameter draws as well as the final loss value (distance from
                reference model)
        """

        # build optimisation framework
        solver = self.architecture
        solver.zero_grad()
        opt = torch.optim.Adam(solver.parameters(), lr=self.learning_rate)

        # run optimisation loop
        for _ in range(self.num_iters):
            opt.zero_grad()
            y_perp = solver(X_perp)
            loss = self.loss.forward(y_ast, y_perp)
            loss.backward()
            opt.step()

        # extract projected parameters and final loss function value
        theta_perp = list(solver.parameters())[0].data
        dist_to_ref_model = loss.item()

        return theta_perp, dist_to_ref_model
