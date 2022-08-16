"""Base projection class."""

from typing import Optional, List, Union

from arviz import InferenceData
from bambi import Model

from kulprit.data.submodel import SubModel
from kulprit.projection.solver import Solver


class Projector:
    def __init__(
        self,
        model: Model,
        idata: InferenceData,
        path: Optional[dict] = None,
        num_steps: Optional[int] = 5_000,
        obj_n_mc: Optional[float] = 10,
    ) -> None:
        """Reference model builder for projection predictive model selection.

        This class handles the core projection methods of the model selection
        procedure. Note that throughout the procedure, variables with names of
        the form ``*_ast`` belong to the reference model while variables with
        names like ``*_perp`` belong to the restricted model. This is to
        preserve notation choices from previous papers on the topic.

        Args:
            data (kulprit.data.ModelData): Reference model dataclass object
            path (dict): An optional search path dictionary, initialised to None
                and assigned by the ReferenceModel parent object following a
                search for efficient submodel retrieval
            num_steps (int): Number of iterations to run VI for
            obj_n_mc (int):
        """

        # log reference model and reference inference data object
        self.idata = idata
        self.model = model

        # set optimiser parameters
        self.num_steps = num_steps
        self.obj_n_mc = obj_n_mc

        # build solver
        self.solver = Solver(
            model=self.model,
            idata=self.idata,
            num_steps=self.num_steps,
            obj_n_mc=self.obj_n_mc,
        )

        # log search path
        self.path = path

    def project(
        self,
        terms: Union[List[str], int],
    ) -> SubModel:
        """Wrapper function for projection method.

        Args:
            terms (Union[List[str], int]): Either a list of strings containing
                the names of the parameters to include the submodel, or the
                number of parameters to include in the submodel, **not**
                including the intercept term

        Returns:
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # project terms by name
        if isinstance(terms, list):
            # perform projection
            return self.project_names(term_names=terms)

        # project a number of terms
        else:
            # test `model_size` input
            if self.path is None or terms not in list(self.path.keys()):
                raise UserWarning(
                    "In order to project onto an integer number of terms, please "
                    + "first complete a parameter search."
                )

            # project onto the search path submodel with `terms` number of terms
            return self.path[terms]

    def project_names(
        self,
        term_names: List[str],
    ) -> SubModel:
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using PyTorch and
        Adam for the optimisation.

        Args:
            term_names (List[str]): The names of parameters to project onto the
                submodel

        Returns:
            kulprit.data.ModelData: Projected submodel ``ModelData`` object
        """

        # solve the parameter projections
        return self.solver.solve(term_names=term_names)
