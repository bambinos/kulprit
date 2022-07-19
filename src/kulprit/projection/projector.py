"""Base projection class."""

from typing import Optional, List, Union

from arviz import InferenceData
from bambi import Model

from kulprit.data.data import ModelData
from kulprit.data.res_idata import init_idata
from kulprit.families.family import Family
from kulprit.projection.solver import Solver


class Projector:
    def __init__(
        self,
        model: Model,
        idata: InferenceData,
        path: Optional[dict] = None,
        num_iters: Optional[int] = 400,
        learning_rate: Optional[float] = 0.01,
        num_thinned_samples: Optional[int] = 400,
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
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): The backprop optimiser's learning rate
        """

        # log reference model and reference inference data object
        self.idata = idata
        self.model = model
        self.family = Family(model)

        # set optimiser parameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.num_thinned_samples = num_thinned_samples

        # build solver
        self.solver = Solver(
            ref_model=self.model,
            ref_idata=self.idata,
            num_iters=self.num_iters,
            learning_rate=self.learning_rate,
        )

        # log search path
        self.path = path

    def project(
        self,
        terms: Union[List[str], int],
    ) -> ModelData:
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
            # test `terms` input
            if not all([term in self.model.term_names for term in terms]):
                raise UserWarning(
                    "Please ensure that all terms selected for projection exist in"
                    + " the reference model."
                )
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
    ) -> ModelData:
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

        # initialise restricted model inference data
        res_idata = init_idata(
            ref_model=self.model,
            ref_idata=self.idata,
            term_names=term_names,
            num_thinned_samples=self.num_thinned_samples,
        )

        # solve the parameter projections
        sub_model = self.solver.solve(res_idata=res_idata, term_names=term_names)
        return sub_model
