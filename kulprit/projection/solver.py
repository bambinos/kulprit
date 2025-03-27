"""optimization module."""
from arviz import from_dict
import numpy as np
from scipy.optimize import minimize


def solve(neg_log_likelihood, pps, initial_guess, var_info, tolerance, rng):
    """The primary projection method in the procedure.

    Parameters:
    ----------
    neg_log_likelihood: Callable
        The negative log-likelihood function of the model
    pps: array
        The predictions of the reference model
    initial_guess: array
        The initial guess for the optimization
    var_info: dict
        The dictionary containing information about the size and transformation of the variables

    Returns:
    -------
        new_idata: arviz.InferenceData
        loss: float
    """
    num_samples, num_obs = pps.shape
    size_rep = max(1, np.log(num_obs).astype(int))
    posterior_array = np.zeros((num_samples, len(initial_guess)))
    posterior_dict = {}
    objectives = []

    opt = minimize(
        neg_log_likelihood,
        args=(pps[-1]),
        x0=initial_guess,
        tol=tolerance,
        method="powell",
    )
    initial_guess = opt.x

    for idx, obs in enumerate(pps):
        rep = rng.choice(range(0, num_obs), size=size_rep, replace=False)
        obs[rep] = pps[idx - 1][rep]
        opt = minimize(
            neg_log_likelihood,
            args=(obs),
            x0=initial_guess,
            tol=tolerance,
            method="powell",
        )

        posterior_array[idx] = opt.x
        objectives.append(opt.fun)
        initial_guess = opt.x

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
    loss = np.mean(objectives)
    return new_idata, loss
