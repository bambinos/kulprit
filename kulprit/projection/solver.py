"""optimization module."""

from arviz import from_dict
import numpy as np
from scipy.optimize import minimize


def solve(
    neg_log_likelihood,
    preds,
    initial_guess,
    var_info,
    weights,
    tolerance,
    active_idx=None,
):
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
        The dictionary containing the size and transformation of the variables
    weights: array or None
        The weights for the clustered predictions, if None, the loss is computed as the mean
        of the objectives, i.e the weights are assumed to be the same for all predictions.
    tolerance: float
        The tolerance for the optimization procedure.
    active_idx: array of int or None
        Flat indices of the parameters that are allowed to be optimized. Excluded
        parameters are kept at their initial values. If None, all parameters are active.

    Returns:
    -------
        new_idata: arviz.InferenceData
        loss: float
    """
    num_samples = len(preds)
    full_dim = len(initial_guess)
    posterior_array = np.zeros((num_samples, full_dim))
    objectives = np.zeros(num_samples)

    if active_idx is None:
        active_idx = np.arange(full_dim)
    else:
        active_idx = np.asarray(active_idx, dtype=int)

    if active_idx.size == 0:
        raise ValueError(
            "No active parameters were selected for optimization. "
            "Check term-to-variable mapping and base terms."
        )

    guess = initial_guess[active_idx].copy()

    def objective_active(active_params, *pred):

        full = initial_guess.copy()
        full[active_idx] = active_params
        return neg_log_likelihood(full, *pred)

    for idx, pred in enumerate(preds):
        if idx == 0:
            tol = tolerance / 1000
        else:
            tol = tolerance
        opt = minimize(
            objective_active,
            args=pred,
            x0=guess,
            method="powell",
            tol=tol,
        )

        full = initial_guess.copy()
        full[active_idx] = opt.x
        posterior_array[idx] = full
        objectives[idx] = opt.fun
        guess = opt.x

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

        new_idata = from_dict({"posterior": posterior_dict})
        loss = np.mean(objectives) * 0.5
    else:
        new_idata = None
        loss = np.sum(np.array(objectives) * weights)
    return new_idata, loss
