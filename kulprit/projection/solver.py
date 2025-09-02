"""optimization module."""

from arviz import from_dict
import numpy as np
from scipy.optimize import minimize


def solve(neg_log_likelihood, preds, initial_guess, var_info, weights, tolerance):
    """The primary projection method in the procedure.

    Parameters:
    ----------
    neg_log_likelihood: Callable
        The negative log-likelihood function of the model
    preds: array
        The predictions of the reference model
    initial_guess: array
        The initial guess for the optimization
    var_info: dict
        The dictionary containing information about the size and transformation of the variables
    weights: array or None
        The weights for the clustered predictions, if None, the loss is computed as the mean
        of the objectives, i.e the weights are assumed to be the same for all predictions.
    tolerance: float
        The tolerance for the optimization procedure.

    Returns:
    -------
        new_idata: arviz.InferenceData
        loss: float
    """
    num_samples = len(preds)
    posterior_array = np.zeros((num_samples, len(initial_guess)))
    objectives = np.zeros(num_samples)

    for idx, pred in enumerate(preds):
        if idx == 0:
            tol = tolerance / 1000
        else:
            tol = tolerance
        opt = minimize(
            neg_log_likelihood,
            args=pred,
            x0=initial_guess,
            method="powell",
            tol=tol,
        )

        posterior_array[idx] = opt.x
        objectives[idx] = opt.fun
        initial_guess = opt.x

    if weights is None:
        posterior_dict = {}
        size = 0
        for key, values in var_info.items():
            shape, new_size, transformation = values
            posterior_dict[key] = posterior_array[:, size : size + new_size].reshape(
                1, num_samples, *shape
            )
            if transformation is not None:
                posterior_dict[key] = transformation(posterior_dict[key])
            size += new_size

        new_idata = from_dict(posterior=posterior_dict)
        loss = np.mean(objectives) * 0.5
    else:
        new_idata = None
        loss = np.sum(np.array(objectives) * weights)
    return new_idata, loss
