"""Submodel dataclass module."""

from dataclasses import dataclass

import arviz
import bambi as bmb


@dataclass
class SubModel:
    """Submodel dataclass.

    Attributes:
        model (bambi.Model): The submodel's associated Bambi model, from which we can
            extract a built pymc model.
        idata (InferenceData): The inference data object of the submodel containing the
            projected posterior draws and log-likelihood.
        size (int): The number of common terms in the model, not including the intercept
        term_names (list): The names of the terms in the model, including the intercept
    """

    model: bmb.models.Model
    idata: arviz.InferenceData
    loss: float
    size: int
    term_names: list
    has_intercept: bool

    def __repr__(self) -> str:
        """String representation of the submodel."""
        if self.has_intercept:
            intercept = ["Intercept"]
        else:
            intercept = []

        return f"{intercept + self.term_names}"
