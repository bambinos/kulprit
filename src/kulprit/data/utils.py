"""Dataclass building utilities module."""

import numpy as np


def _posterior_to_points(posterior, ref_model):
    """Convert the posterior samples from a restricted model into list of dicts.

    This list of dicts datatype is referred to a `points` in PyMC, and is needed
    to be able to compute the log-likelihood of a projected model, here
    `res_model`.

    Args:
        posterior (dict): Dictionary of posterior restricted model samples
        ref_model (kulprit.ModelData): The reference model to act as a guide

    Returns:
        list: The list of dictionaries of point samples
    """

    # build samples dictionary from posterior of idata
    samples = {
        key: (
            posterior[key].flatten()
            if key in posterior.keys()
            else np.zeros((ref_model.num_draws,))
        )
        for key in ref_model.backend.model.test_point.keys()
    }
    # extract observed and unobserved RV names and sample matrix
    var_names = list(samples.keys())
    obs_matrix = np.vstack(list(samples.values()))
    # build points list of dictionaries
    points = [
        {
            var_names[j]: (
                np.array([obs_matrix[j, i]])
                if var_names[j] != f"{ref_model.response_name}_sigma_log__"
                else np.array(obs_matrix[j, i])
            )
            for j in range(obs_matrix.shape[0])
        }
        for i in range(obs_matrix.shape[1])
    ]
    return points
