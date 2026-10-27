"""optimization module."""

from arviz import from_dict
import numpy as np
from scipy.optimize import minimize


def _solve_blocks(blocks, grad=None, shift=0.0):
    """Newton step and log-determinant for a block-diagonal Hessian.

    Parameters
    ----------
    blocks : list of (index, matrices) pairs
        Each ``matrices`` is a stack of symmetric blocks that act on ``grad[index]``.
    grad : array_like, optional
        The gradient corresponding to the blocks. If ``None``, the step is not computed.
    shift : float, optional
        A shift added to the diagonal of each block to ensure positive definiteness.

    Returns
    -------
    step : array_like or None
        The Newton step corresponding to the blocks, or ``None`` if ``grad`` is ``None``.
    logdet : float
        The log-determinant of the block-diagonal Hessian.
    """
    step = None if grad is None else np.empty_like(grad)
    logdet = 0.0
    for idx, mats in blocks:
        mats = mats + shift * np.eye(mats.shape[-1])
        chol = np.linalg.cholesky(mats)
        logdet += 2.0 * float(np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1))))
        if grad is not None:
            step[idx] = -np.linalg.solve(mats, grad[idx][..., None])[..., 0]
    return step, logdet


def _blocks_logdet(blocks):
    """Log-determinant of the block Hessian, or ``None`` if not positive definite or finite."""
    try:
        logdet = _solve_blocks(blocks)[1]
    except np.linalg.LinAlgError:
        return None
    return logdet if np.isfinite(logdet) else None


def _newton_inner(objective, full, inner_idx, pred, max_iter=50, decrement_tol=1e-6):
    """Damped Newton minimization of the objective over the inner parameters.

    Parameters
    ----------
    objective : callable
        The objective function to minimize. Should return the value, gradient, and block Hessians.
    full : array_like
        The full parameter vector, including both inner and outer parameters.
    inner_idx : array_like
        Indices of the inner parameters to optimize.
    pred : array_like
        Predictor variables or other fixed inputs to the objective function.
    max_iter : int, optional
        Maximum number of Newton iterations.
    decrement_tol : float, optional
        Tolerance for the Newton decrement to determine convergence.

    Returns
    -------
    z : array_like
        The optimized inner parameters.
    value : float
        The value of the objective function at the optimum.
    logdet : float or None
        The log-determinant of the block Hessian at the optimum, or ``None``
        if not positive definite.
    """

    def make_trial(z):
        trial = full.copy()
        trial[inner_idx] = z
        return trial

    z = full[inner_idx].copy()
    value, grad, blocks = objective.system(make_trial(z), pred)
    for _ in range(max_iter):
        if not (np.isfinite(grad).all() and all(np.isfinite(mats).all() for _, mats in blocks)):
            break
        shift = 0.0
        while True:
            try:
                step, logdet = _solve_blocks(blocks, grad, shift)
                break
            except np.linalg.LinAlgError:
                scale = max(float(np.abs(mats).max()) for _, mats in blocks)
                shift = max(2.0 * shift, 1e-3 * scale, 1e-8)
        slope = float(grad @ step)
        if shift == 0.0 and -0.5 * slope < decrement_tol:
            return z, value, logdet if np.isfinite(logdet) else None
        scale = 1.0
        while True:
            candidate = z + scale * step
            new_value, new_grad, new_blocks = objective.system(make_trial(candidate), pred)
            if np.isfinite(new_value) and new_value <= value + 1e-4 * scale * slope:
                break
            scale *= 0.5
            if scale < 1e-8:
                return z, value, _blocks_logdet(blocks)
        z, value, grad, blocks = candidate, new_value, new_grad, new_blocks
    return z, value, _blocks_logdet(blocks)


def _laplace_value(neg_phi, logdet, n_inner):
    """Laplace-corrected negative marginal log-probability of the inner parameters.

    Parameters
    ----------
    neg_phi : float
        The negative log-probability of the inner parameters at the mode.
    logdet : float or None
        The log-determinant of the inner block of the Hessian at the mode.
    n_inner : int
        The number of inner parameters.

    Returns
    -------
    float
        The Laplace-corrected negative marginal log-probability of the inner parameters.
        If the log-determinant is not finite, returns infinity.
    """
    if logdet is None or not np.isfinite(logdet):
        return np.inf
    return float(neg_phi) + 0.5 * logdet - 0.5 * n_inner * np.log(2.0 * np.pi)


def solve(
    objective,
    preds,
    initial_guess,
    var_info,
    weights,
    tolerance,
    active_idx=None,
    inner_idx=None,
):
    """The primary projection method in the procedure.

    Parameters:
    ----------
    objective: Callable or ProbedMarginalObjective
        The objective function to be minimized. With ``inner_idx`` it must provide
        ``system(params, pred)`` returning the value, the inner gradient and the inner
        Hessian blocks (see ``ProbedMarginalObjective``).
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
    inner_idx: array of int or None
        Flat indices of the group-effect parameters profiled out with the Laplace
        correction; disjoint from ``active_idx``. If None or empty, the objective is
        optimized directly.

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

    if inner_idx is None:
        inner_idx = np.array([], dtype=int)
    else:
        inner_idx = np.asarray(inner_idx, dtype=int)

    use_laplace = inner_idx.size > 0
    inner_state = initial_guess[inner_idx].copy()

    guess = initial_guess[active_idx].copy()

    def profile(active_params, pred):
        """Laplace value of the inner-profiled objective, updating the inner warm start."""
        full = initial_guess.copy()
        full[inner_idx] = inner_state
        full[active_idx] = active_params
        b_hat, value, logdet = _newton_inner(objective, full, inner_idx, pred)
        inner_state[:] = b_hat
        full[inner_idx] = b_hat
        return full, _laplace_value(value, logdet, inner_idx.size)

    def objective_active(active_params, pred):
        """Objective function restricted to the active parameters."""
        if use_laplace:
            return profile(active_params, pred)[1]
        full = initial_guess.copy()
        full[active_idx] = active_params
        return objective(full, *(pred if isinstance(pred, tuple) else (pred,)))

    for idx, pred in enumerate(preds):
        if idx == 0:
            tol = tolerance / 1000
        else:
            tol = tolerance
        opt = minimize(
            objective_active,
            args=(pred,),
            x0=guess,
            method="powell",
            tol=tol,
        )

        if use_laplace:
            full, objectives[idx] = profile(opt.x, pred)
        else:
            full = initial_guess.copy()
            full[active_idx] = opt.x
            objectives[idx] = opt.fun
        posterior_array[idx] = full
        guess = opt.x

    if weights is None:
        posterior_dict = {}
        size = 0
        for key, values in var_info.items():
            shape, new_size, transformation, _ = values
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
