"""Module for handling data throughout the procedure."""

import dataclasses

import torch
from arviz import InferenceData
from .structure import ModelStructure


@dataclasses.dataclass(order=True)
class ModelData:
    """Data class for handling model data.

    This class serves as the primary data container passed throughout the
    procedure, allowing for more simple and legible code. Note that this class
    supports ordering, and we choose distance to reference model as our sorting
    index. Naturally, this value is set to zero for the reference model,
    providing a known global minimum.

    Attributes:
        structure (kulprit.data.ModelStructureData): ModelStructureData object
            built from a Bambi model
        idata (arviz.InferenceData): InferenceData object of the fitted Bambi
            model
        dist_to_ref_model (torch.tensor): The distance from this model to the
            reference model being used in the procedure as defined by the loss
            function
        sort_index (int): Sorting index attribute used in forward search method
    """

    structure: ModelStructure
    idata: InferenceData
    dist_to_ref_model: torch.tensor
    sort_index: int = dataclasses.field(init=False)

    def __post_init__(self):
        # use the distance from the reference model as the ordering index
        self.sort_index = self.dist_to_ref_model

        # add MCMC draws dimension to the structure object
        self.structure.num_draws = (
            self.idata.posterior.dims["chain"] * self.idata.posterior.dims["draw"]
        )

        if self.structure.predictions is None:
            # make insample predictions for the model and append to structure
            self.structure.predictions = self.structure.predict(
                idata=self.idata, inplace=False, kind="pps"
            ).posterior_predictive[self.structure.response_name]
