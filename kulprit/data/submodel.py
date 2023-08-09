"""Submodel dataclass module."""

from dataclasses import dataclass

import arviz
import bambi


@dataclass
class SubModel:
    """Submodel dataclass.

    Attributes:
        model (bambi.Model): The submodel's associated Bambi model, from which we can
            extract a built pymc model.
        idata (InferenceData): The inference data object of the submodel containing the
            projected posterior draws and log-likelihood.
        loss (float): The final loss (negative log-likelihood) of the submodel following
            projection predictive inference
        size (int): The number of common terms in the model, not including the intercept
        term_names (list): The names of the terms in the model, including the intercept
    """

    model: bambi.models.Model
    idata: arviz.InferenceData
    loss: float
    size: int
    term_names: list

    def __repr__(self) -> str:
        """String representation of the submodel."""

        return ", ".join([self.model.formula.main] + list(self.model.formula.additionals))
