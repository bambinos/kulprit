"""Functions to interact with PyMC models"""

import warnings
import numpy as np
import pytensor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pymc import do, compute_log_likelihood
from pymc.logprob.utils import ParameterValueError
from pymc.util import is_transformed_name, get_untransformed_name
from pymc.pytensorf import join_nonshared_inputs
from pytensor import function, shared
from pytensor.tensor import tensor
from pytensor.graph import ancestors


def compile_mllk(model, initial_point):
    """
    Compile the log-likelihood function for the model to be able to condition on both
    data and parameters.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model for which to compile the log-likelihood.
    initial_point : dict
        A dictionary mapping model variables to their initial values.

    Returns
    -------
    fmodel : callable
        A function that takes parameters and pseudo-observations and returns
        the negative log-likelihood.
    """
    obs_rvs = model.observed_RVs[0]
    new_y_value = obs_rvs.type()
    original_value = model.rvs_to_values.get(obs_rvs)
    model.rvs_to_values[obs_rvs] = new_y_value

    vars_ = model.value_vars

    try:
        [logp], raveled_inp = join_nonshared_inputs(
            point=initial_point, outputs=[model.datalogp], inputs=vars_
        )
        rv_logp_fn = function([raveled_inp, new_y_value], logp)
        rv_logp_fn.trust_input = True

        def fmodel(params, *pred):
            try:
                if len(pred) == 2:
                    return -(rv_logp_fn(params, pred[0]) + rv_logp_fn(params, pred[1]))
                return -(rv_logp_fn(params, pred[0]))
            except ParameterValueError:
                # Handle floating-point edge cases underflow/overflow for extreme values on the
                #  transformed unbounded scale.
                return np.inf

        return fmodel
    finally:
        if original_value is None:
            del model.rvs_to_values[obs_rvs]
        else:
            model.rvs_to_values[obs_rvs] = original_value


def _as_rows(pred):
    """Flatten pseudo-observation rows into a tuple of single arrays.

    Parameters
    ----------
    pred : array or tuple of arrays
        The pseudo-observation rows to be flattened.

    Returns
    -------
    tuple of arrays
        The flattened pseudo-observation rows.
    """
    if isinstance(pred, np.ndarray):
        return (pred,)
    rows = []
    for item in pred:
        if isinstance(item, (tuple, list)):
            rows.extend(item)
        else:
            rows.append(item)
    return tuple(rows)


_PRIOR_ON = np.array(1.0)
_PRIOR_OFF = np.array(0.0)


class _LazyFunction:
    """Compile a PyTensor function on first call.

    Each variant of the objective is only compiled if it is actually used.
    """

    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
        self._compiled = None

    def __call__(self, *args):
        if self._compiled is None:
            self._compiled = function(self._inputs, self._outputs, on_unused_input="ignore")
            self._compiled.trust_input = True
        return self._compiled(*args)


class MarginalObjective:
    """Negative log pseudo-likelihood plus group-effect priors, with derivatives.

    Only group-effect priors are included, so the remaining parameters are fitted by
    maximum likelihood. ``dense`` returns value, gradient and full Hessian;
    ``probe`` returns value, gradient and Hessian-vector products along given
    directions. Variants compile lazily. A pseudo-observation is one array or a
    tuple of rows; the prior is counted once.

    Parameters
    ----------
    dense : callable
        Function returning value, gradient and full Hessian.
    make_probe : callable or None
        Function returning a probe for Hessian-vector products, or None if not available.

    Returns
    -------
    MarginalObjective
        An object representing the marginal objective with dense and probe variants.
    """

    def __init__(self, dense, make_probe=None):
        self._dense = dense
        self._make_probe = make_probe
        self._probes = {}

    @property
    def has_probe(self):
        """True when the Hessian-vector probe variant is available."""
        return self._make_probe is not None

    @staticmethod
    def _accumulate(function_, params, pred, *extra):
        total = None
        for k, row in enumerate(_as_rows(pred)):
            out = function_(params, row, _PRIOR_ON if k == 0 else _PRIOR_OFF, *extra)
            total = out if total is None else [acc + new for acc, new in zip(total, out)]
        return total

    def dense(self, params, pred):
        """Value, gradient and dense Hessian at ``params``."""
        return self._accumulate(self._dense, params, pred)

    def probe(self, params, pred, directions):
        """Value, gradient and Hessian products along the columns of ``directions``."""
        n_directions = directions.shape[1]
        if n_directions not in self._probes:
            self._probes[n_directions] = self._make_probe(n_directions)
        return self._accumulate(self._probes[n_directions], params, pred, *directions.T)


def _blocks_from_support(mat, rel_eps=1e-10):
    """Index arrays of the connected components of a Hessian support pattern."""
    threshold = rel_eps * max(float(np.abs(mat).max()), 1e-300)
    _, labels = connected_components(csr_matrix(np.abs(mat) > threshold), directed=False)
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.cumsum(np.bincount(labels))[:-1])


def _blocks_to_dense(blocks, size):
    """Scatter ``(index, matrices)`` blocks into a dense ``size`` x ``size`` matrix."""
    dense = np.zeros((size, size))
    for idx, mats in blocks:
        dense[idx[:, :, None], idx[:, None, :]] = mats
    return dense


class ProbedMarginalObjective:
    """Inner-parameter system (value, gradient, Hessian blocks) for the Newton solve.

    The inner Hessian is block-diagonal by group level. Giving the ``r``-th
    parameter of every block to probe direction ``r`` recovers all blocks from as
    many Hessian-vector products as the largest block, which is cheaper than a dense
    Hessian. ``system`` returns the Hessian as ``(index, matrices)`` pairs, one per
    block size.

    Blocks come from the support of the dense Hessian at a calibration point. The
    dense Hessian (a single block) is used instead when a block exceeds
    ``MAX_BLOCK_SIZE`` (e.g. crossed effects), there are too few inner parameters, or
    the recovered blocks do not match the dense Hessian at validation points.
    """

    MIN_INNER_FOR_PROBE = 32
    MAX_BLOCK_SIZE = 4

    def __init__(self, base, inner_idx, start, calibration_row, extra_row=None, seed=1234):
        self._base = base
        self._inner_idx = np.asarray(inner_idx, dtype=int)
        self._usable = False
        self._directions = None
        self._groups = None
        if not base.has_probe or self._inner_idx.size <= self.MIN_INNER_FOR_PROBE:
            return
        try:
            inner_idx = self._inner_idx
            dim = np.size(start)
            calibration_rows = (calibration_row,)
            extra_rows = (extra_row,) if extra_row is not None else None
            calibration_hess = base.dense(start, calibration_rows)[2][np.ix_(inner_idx, inner_idx)]
            blocks = _blocks_from_support(calibration_hess)
            max_size = max(len(block) for block in blocks)
            if max_size > self.MAX_BLOCK_SIZE:
                return
            self._directions = np.zeros((dim, max_size))
            for block in blocks:
                self._directions[inner_idx[block], np.arange(len(block))] = 1.0
            by_size = {}
            for block in blocks:
                by_size.setdefault(len(block), []).append(block)
            self._groups = [np.array(group) for group in by_size.values()]
            rng = np.random.default_rng(seed)
            checks = [(start, calibration_rows)]
            if extra_rows is not None:
                checks.append((start, extra_rows))
            checks.append((start + 0.05 * rng.normal(size=dim), calibration_rows))
            for params, rows in checks:
                recovered = _blocks_to_dense(self._probed(params, rows)[2], inner_idx.size)
                exact = base.dense(params, rows)[2][np.ix_(inner_idx, inner_idx)]
                scale = max(float(np.abs(exact).max()), 1e-300)
                if float(np.abs(recovered - exact).max()) / scale > 1e-6:
                    return
            self._usable = True
        except (
            ParameterValueError,
            np.linalg.LinAlgError,
            ValueError,
            TypeError,
            NotImplementedError,
        ):
            self._usable = False

    @property
    def usable(self):
        """True when probe-based recovery passed validation."""
        return self._usable

    def _probed(self, params, pred):
        value, grad, columns = self._base.probe(params, pred, self._directions)
        columns = columns[self._inner_idx]
        # element r of each block sits in column r, so block[a, c] = columns[idx[a], c]
        blocks = [(idx, columns[idx][:, :, : idx.shape[1]]) for idx in self._groups]
        return value, grad, blocks

    def system(self, params, pred):
        """Value, inner gradient and symmetrized inner Hessian blocks at ``params``."""
        inner_idx = self._inner_idx
        n_inner = inner_idx.size
        try:
            if self._usable:
                value, grad, blocks = self._probed(params, pred)
            else:
                value, grad, hess = self._base.dense(params, pred)
                blocks = [(np.arange(n_inner)[None], hess[np.ix_(inner_idx, inner_idx)][None])]
        except ParameterValueError:
            # Handle floating-point edge cases underflow/overflow for extreme values on
            # the transformed unbounded scale.
            identity = [(np.arange(n_inner)[:, None], np.ones((n_inner, 1, 1)))]
            return np.inf, np.zeros(n_inner), identity
        blocks = [(idx, 0.5 * (mats + mats.transpose(0, 2, 1))) for idx, mats in blocks]
        return float(value), grad[inner_idx], blocks


def compile_marginal_mllk(model, initial_point, term_partition, switches):
    """
    Compile the projection objective for a model with group-specific terms.

    The objective is the negative of ``log p(pseudo | theta) + sum_g log p(coef_g |
    sigma_g)``. Group-effect priors are gated by the term switches, so switched-off
    terms contribute nothing. Returns a ``MarginalObjective``.

    Parameters
    ----------
    model : pymc.Model
        The switched reference model.
    initial_point : dict
        The model's initial point.
    term_partition : dict
        Term names to ``(coefficient_vars, sigma_vars)``, from ``get_term_partition``.
    switches : dict
        Term names to their shared switch variables.

    Returns
    -------
    MarginalObjective
        An object representing the marginal objective with dense and probe variants.
    """
    obs_rvs = model.observed_RVs[0]
    new_y_value = obs_rvs.type()
    original_value = model.rvs_to_values.get(obs_rvs)
    model.rvs_to_values[obs_rvs] = new_y_value

    try:
        # sum the priors of the group-effect coefficient/offset RVs, gated per term
        free_rvs = {rv.name: rv for rv in model.free_RVs}
        prior_lp = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Intervention expression references")
            for term_name in term_partition:
                if "|" not in term_name:
                    continue
                coefficient_names = sorted(term_partition[term_name][0])
                if not coefficient_names:
                    continue
                term_lp = model.logp(
                    vars=[free_rvs[name] for name in coefficient_names], jacobian=True, sum=True
                )
                term_lp = term_lp * switches[term_name]
                prior_lp = term_lp if prior_lp is None else prior_lp + term_lp

        outputs = [model.datalogp]
        if prior_lp is not None:
            outputs.append(prior_lp)
        joined, raveled_inp = join_nonshared_inputs(
            point=initial_point, outputs=outputs, inputs=model.value_vars
        )

        # the prior weight lets a pair of pseudo-observation rows count the prior once
        prior_weight = pytensor.tensor.scalar(dtype="float64")
        cost = -joined[0]
        if prior_lp is not None:
            cost = cost - prior_weight * joined[1]
        grad = pytensor.grad(cost, raveled_inp, disconnected_inputs="ignore")
        inputs = [raveled_inp, new_y_value, prior_weight]
        dense = _LazyFunction(inputs, [cost, grad, pytensor.gradient.hessian(cost, raveled_inp)])

        def make_probe(n_directions):
            directions = [pytensor.tensor.vector() for _ in range(n_directions)]
            columns = []
            for direction in directions:
                column = pytensor.gradient.pushforward(
                    grad, raveled_inp, tangents=direction, disconnected_outputs="ignore"
                )
                if isinstance(column, (list, tuple)):
                    column = column[0]
                columns.append(column)
            return _LazyFunction(
                [*inputs, *directions], [cost, grad, pytensor.tensor.stack(columns, axis=1)]
            )

    finally:
        if original_value is None:
            del model.rvs_to_values[obs_rvs]
        else:
            model.rvs_to_values[obs_rvs] = original_value

    return MarginalObjective(dense, make_probe if prior_lp is not None else None)


def turn_off_terms(switches, all_terms, term_names):
    """
    Turn off the terms not in term_names
    """
    for term in all_terms:
        if term not in term_names:
            switches[term].set_value(0.0)
        else:
            switches[term].set_value(1.0)


def add_switches(model, ref_terms):
    switches = {term: shared(1.0) for term in ref_terms}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Intervention expression references")
        switched_terms = {term: model.named_vars[term] * switches[term] for term in ref_terms}
        return do(model, switched_terms), switches


def compute_llk(idata, model):
    """Compute log-likelihood for the submodel."""
    return compute_log_likelihood(idata, model=model, progressbar=False, extend_inferencedata=False)


def get_term_partition(model, term_names):
    """
    Map each switchable term name to ``(coefficient_vars, sigma_vars)``.

    ``coefficient_vars`` are the free RVs carrying the term's coefficients (the
    coefficient or, if non-centered, the offset RV). ``sigma_vars`` (group terms only)
    are ``{term}_sigma`` and the free RVs feeding it. Sigma is kept out of the
    coefficient set because the likelihood does not identify it jointly with them.
    """
    free_rv_names = {rv.name for rv in model.free_RVs}
    partition = {}
    for term_name in term_names:
        if term_name not in model.named_vars:
            raise KeyError(
                f"Term '{term_name}' is not available in model.named_vars. "
                "This can happen when term aliases do not match backend variable names."
            )
        term_var = model.named_vars[term_name]
        anc_names = {var.name for var in ancestors([term_var])}
        # include the term variable itself when it is a free RV (centered case)
        if term_var.name in free_rv_names:
            anc_names.add(term_var.name)
        vars_ = anc_names & free_rv_names
        sigma_vars = set()
        if "|" in term_name:
            sigma_name = f"{term_name}_sigma"
            if sigma_name in free_rv_names:
                sigma_var = model.named_vars[sigma_name]
                sigma_vars = {var.name for var in ancestors([sigma_var])} & free_rv_names
                sigma_vars.add(sigma_name)
                vars_ = vars_ - sigma_vars
        if not vars_:
            raise ValueError(
                f"Term '{term_name}' does not map to any optimizable free RV after filtering."
            )
        partition[term_name] = (vars_, sigma_vars)
    return partition


def get_term_variables(model, term_names):
    """Map each switchable term name to its coefficient free-RV names (sigma excluded)."""
    return {
        term: coefficient_vars
        for term, (coefficient_vars, _) in get_term_partition(model, term_names).items()
    }


def get_model_information(model, initial_point):
    """
    Get size and transformations of each variable in a PyMC model.

    ``transformation`` maps the unconstrained scale to the original scale (backward),
    ``inverse`` the reverse (forward). Both are compiled plain callables.
    """

    var_info = {}
    for v_var in model.value_vars:
        name = v_var.name
        if is_transformed_name(name):
            name = get_untransformed_name(name)
            ndim = initial_point[v_var.name].ndim
            # PyTensor may interpret names starting with digits or containing '|' as dtype
            # strings, so sanitize the variable name.
            safe_name = name.replace("|", "_").replace(":", "_").replace(" ", "_")
            rv = model.values_to_rvs[v_var]
            transform = model.rvs_to_transforms[rv]
            x_var = tensor(
                dtype="float64", shape=(None,) * (ndim + 2), name=f"transformed_{safe_name}"
            )
            y_var = tensor(dtype="float64", shape=(None,) * (ndim + 2), name=f"inverse_{safe_name}")
            transformation = function(
                inputs=[x_var], outputs=transform.backward(x_var, *rv.owner.inputs)
            )
            inverse = function(inputs=[y_var], outputs=transform.forward(y_var, *rv.owner.inputs))
        else:
            transformation = None
            inverse = None

        var_info[name] = (
            initial_point[v_var.name].shape,
            initial_point[v_var.name].size,
            transformation,
            inverse,
        )

    return var_info
