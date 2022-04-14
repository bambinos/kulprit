"""Utility functions module."""

import torch


def _extract_insample_predictions(model):
    """Extract some model's in-sample predictions.

    To do:
        * consider moving this method either into the base class, or into a
            separate utility function collection for sparsity

    Args:
        model (kulprit.ModelData): Some model we wish to get predictions from

    Returns:
        torch.tensor: The in-sample predictions
    """

    y_pred = torch.from_numpy(
        model.predictions.stack(samples=("chain", "draw")).values.T
    ).float()
    return y_pred


def _compute_elpd(model):
    """Compute the ELPD LOO estimates from a fitted model.

    Args:
        model (kulprit.ModelData): The model to diagnose

    Returns:
        arviz.ELPDData: The ELPD LOO estimates
    """

    raise NotImplementedError
