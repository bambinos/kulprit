"""Module for handling reference model data."""

import dataclasses
from bambi import Model

import torch
from arviz import InferenceData

from kulprit.families.family import Family


@dataclasses.dataclass(order=True)
class ReferenceModelData:
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

    model: Model
    idata: InferenceData

    def __post_init__(self):
        # define model family
        self.family = Family(
            name=self.model.family.name,
            link=self.model.family.link.name,
            data=self.model,
        )

        # add MCMC draws dimension to the structure object
        self.num_draws = (
            self.idata.posterior.dims["chain"] * self.idata.posterior.dims["draw"]
        )
        self.num_obs, self.num_covs = self.model._design.common.shape
