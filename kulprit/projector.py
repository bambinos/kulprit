# pylint: disable=undefined-loop-variable
# pylint: disable=too-many-instance-attributes
"""Core reference model class."""
import warnings
from copy import copy
import numpy as np
from pandas import DataFrame

from bambi import formula

from kulprit.projection.arviz_io import check_idata, compute_loo, get_observed_data, get_pps
from kulprit.projection.pymc_io import (
    add_switches,
    compile_mllk,
    compute_llk,
    turn_off_terms,
    get_model_information,
)
from kulprit.projection.search_strategies import (
    user_path,
    forward_search,
    l1_search,
    _missing_lower_order_terms,
)
from kulprit.projection.solver import solve


class ProjectionPredictive:
    """
    Projection Predictive class from which we perform the model selection procedure.

    Parameters:
    ----------
    model : Bambi model
        The reference model to project
    idata : InferenceData or DataTree
        The result of fitting reference model
    rng : RandomState
        Random number generator used for sampling from the posterior predictive if
        the group is not present in idata.
    """

    def __init__(self, model, idata, rng=456):
        """Builder for projection predictive model selection."""
        # initialize attributes
        self.num_samples = None
        self.num_clusters = None
        self.early_stop = None
        self.tolerance = None
        self._pps = None
        self._ppc = None
        self._weights = None
        self._list_of_submodels = []

        # set random number generator
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(rng)

        # check we have the model fitted
        if not model.built:
            raise ValueError(
                "Before projecting, please fit the model, using the `fit` method.\n"
                "Additionally, make sure that the sampling converged, "
                "and that the model fits well the data."
            )

        # get information from Bambi's reference model
        self._has_intercept = formula.formula_has_intercept(model.formula.main)
        self._response_name = model.response_component.term.name
        self._ref_terms = [
            v.alias if v.alias is not None else k
            for k, v in model.components[model.family.likelihood.parent].common_terms.items()
        ]
        self._categorical_terms = sum(
            term.categorical
            for term in model.components[model.family.likelihood.parent].common_terms.values()
        )

        self._base_terms = _get_base_terms(self._has_intercept, model.constant_components)

        # get information from PyMC's reference model
        self._pymc_model = copy(model.backend.model)
        initial_point = self._pymc_model.initial_point()
        self._ref_var_info = get_model_information(self._pymc_model, initial_point)
        self._initial_guess = np.concatenate([np.ravel(value) for value in initial_point.values()])

        # add switches to the model to turn on/off terms in the model
        # without having to rebuild the model
        self._pymc_model_sw, self._switches = add_switches(self._pymc_model, self._ref_terms)
        self._neg_log_likelihood = compile_mllk(self._pymc_model_sw, initial_point)

        # get information from ArviZ's InferenceData object
        idata = check_idata(idata, model, self._rng)
        self._observed_dataset, self._observed_array = get_observed_data(idata, self._response_name)

        elpd_ref = compute_loo(idata=idata)

        self.reference_model = RefModel(
            model=model,
            idata=idata,
            elpd=elpd_ref.elpd,
            elpd_se=elpd_ref.se,
            term_names=[fvar.name for fvar in self._pymc_model.free_RVs],
        )

    def __repr__(self) -> str:
        """Return the terms of the submodels."""

        if not self._list_of_submodels:
            return "ReferenceModel"

        else:
            str_of_submodels = "\n".join(
                f"{idx:>3} " f"{value.term_names}"
                for idx, value in enumerate(self._list_of_submodels)
            )
            return str_of_submodels

    def __iter__(self):
        """Iterate over the submodels."""
        yield from self._list_of_submodels

    def __len__(self):
        """Return the number of submodels."""
        return len(self._list_of_submodels)

    def __getitem__(self, index):
        """Return submodels by index or slice."""

        if isinstance(index, int):
            return self._list_of_submodels[index]
        else:
            n_submodels = len(self._list_of_submodels)
            if not all(-n_submodels <= i < n_submodels for i in index):
                warnings.warn(
                    "At least one index is out of bounds. Ignoring out of bounds indices."
                )
                index = [i for i in index if -n_submodels <= i < n_submodels]
            return [self._list_of_submodels[i] for i in index]

    def project(
        self,
        method="forward",
        user_terms=None,
        num_samples=400,
        num_clusters=20,
        early_stop=None,
        require_lower_terms=True,
        tolerance=1,
    ):
        """Perform model projection.

        Parameters:
        -----------
        method : str
            The search method to employ, either "forward" for a forward search, or "l1" for
            an L1-regularized search. Ignored if "user_terms" is provided.
        user_terms : list of list of str
            If a nested list of terms is provided, model with those terms will be projected
            directly.
        num_samples : int
            The number of samples to draw from the posterior predictive distribution for the
            projection procedure and ELPD computation. Defaults to 400.
        num_clusters : int
            The number of clusters to use during the forward search. Defaults to 20.
        If None, the number of clusters is set to the number of samples.
        If num_clusters is larger than num_samples, it is set to num_samples.
        early_stop : str or int, optional
            Whether to stop the search earlier. If an integer is provided, the search stops
            when the submodel size is equal to the integer. If a string is provided, the search
            stops when the difference in ELPD between the reference and submodel is small.
            There are two criteria to define what small is, "mean" and "se".
            The "mean" criterion stops the search when the difference between a the ELPD is smaller
            than 4. The "se" criterion stops the search when the ELPD of the submodel is within
            one standard error of the reference model. Defaults to None.
        require_lower_terms : bool
            Include higher-order interactions only if all lower-order interactions and main effects
            are already in the subset. Defaults to True. Ignored if user_terms is provided or
            if the method is not "forward".
        tolerance : float
            The tolerance for the optimization procedure. Defaults to 1. Decreasing this value
            will increase the accuracy of the projection at the cost of speed.
        """
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.early_stop = early_stop
        self.tolerance = tolerance
        self._pps, self._ppc, self._weights = get_pps(
            self.reference_model.idata,
            self._response_name,
            self.num_samples,
            self.num_clusters,
            self._rng,
        )

        _check_interactions(self._ref_terms, method, require_lower_terms)

        # if user provided the terms we used them directly, no search is performed
        if user_terms is not None:
            # check if the terms are a list of lists
            if not isinstance(user_terms, list) or not all(
                isinstance(term, list) for term in user_terms
            ):
                raise ValueError("Please provide a list of lists of terms.")
            # check if the length of the submodels always increase
            if not all(
                len(user_terms[idx]) < len(user_terms[idx + 1])
                for idx in range(len(user_terms) - 1)
            ):
                raise ValueError("Please provide a list of terms in increasing order")
            # check if the listed terms are valid
            for idx, term_names in enumerate(user_terms):
                if not set(term_names).issubset(self.reference_model.term_names):
                    raise ValueError(f"Term {idx} is not a valid term in the reference")

            self._list_of_submodels = user_path(self._project, user_terms)
        else:
            if method not in ["forward", "l1"]:
                raise ValueError("Please select either forward search or L1 search.")

            max_terms = len(
                self.reference_model.bambi_model.components[
                    self.reference_model.bambi_model.family.likelihood.parent
                ].common_terms
            )

            if isinstance(self.early_stop, int):
                if self.early_stop < 0:
                    raise ValueError("The early stopping value must be a positive integer.")

                if self.early_stop > max_terms:
                    warnings.warn(
                        "early_stop is larger than the number of terms in the reference model."
                        + "Searching for {n_terms}."
                    )
                    self.early_stop = max_terms
                max_terms = self.early_stop

            if method == "forward":
                self._list_of_submodels = forward_search(
                    self._project,
                    self._ref_terms,
                    max_terms,
                    self.reference_model.elpd,
                    self.early_stop,
                    requiere_lower_terms=require_lower_terms,
                )
            else:
                # currently L1 search is not implemented for categorical models
                if self._categorical_terms:
                    raise NotImplementedError("Group-lasso not yet implemented")

                self._list_of_submodels = l1_search(
                    self._project,
                    self.reference_model.bambi_model,
                    self._ref_terms,
                    max_terms,
                    self.reference_model.elpd,
                    self.early_stop,
                )

    def select(self, criterion="mean"):
        """Select the smallest submodel

        The selection is based on comparing the ELPDs of the reference and submodels.

        Parameters
        ----------
        criterion : str
            The criterion to use for selecting the best submodel. Either "mean" or "se".
            The "mean" criterion selects the smallest submodel with an ELPD that is within
            4 units of the reference model. The "se" criterion selects the smallest submodel
            with an ELPD that is within one standard error of the reference model.

        Returns
        -------
        SubModel
            The selected submodel.
        """
        if criterion not in ["mean", "se"]:
            raise ValueError("Please select either mean or se as the methods.")

        for submodel in self._list_of_submodels:
            if criterion == "mean":
                if (self.reference_model.elpd - submodel.elpd) < 4:
                    return submodel
            else:
                if submodel.elpd + submodel.elpd_se >= self.reference_model.elpd:
                    return submodel

        msg = ""
        if isinstance(self.early_stop, int) and self.early_stop < len(self._ref_terms):
            msg = (
                f"`early_stop` has been set to {self.early_stop}, "
                "try using a larger value or `None`."
            )
        elif isinstance(self.early_stop, str):
            msg = (
                f"`early_stop` has been set to {self.early_stop}, "
                "try using a different criterion, an integer, or `None`."
            )

        warnings.warn(
            "No model has been selected.\n"
            "Use `compare` and `plot_compare()` to identify the problem."
            f"\n{msg}"
        )

        return None

    def _project(self, term_names, clusters=True):
        turn_off_terms(self._switches, self._ref_terms, term_names)

        if clusters:
            samples = self._ppc
            weights = self._weights
        else:
            samples = self._pps
            weights = None

        new_idata, loss = solve(
            self._neg_log_likelihood,
            samples,
            self._initial_guess,
            self._ref_var_info,
            weights,
            self.tolerance,
        )

        # Add observed data and log-likelihood to the projected InferenceData object
        # We only do this for the selected projected model, not the intermediate ones
        if new_idata is not None:
            new_idata.add_groups(observed_data=self._observed_dataset)
            new_idata.add_groups(log_likelihood=compute_llk(new_idata, self._pymc_model))
            # remove the variables that are not in the submodel
            vars_to_drop = [
                var
                for var in new_idata.posterior.data_vars
                if var not in (term_names + self._base_terms)
            ]
            new_idata.posterior = new_idata.posterior.drop_vars(vars_to_drop)

        # build SubModel object and return
        sub_model = SubModel(
            model=self._pymc_model_sw,
            idata=new_idata,
            loss=loss,
            elpd=None,
            elpd_se=None,
            size=len(term_names),
            term_names=term_names,
            has_intercept=self._has_intercept,
        )
        return sub_model

    def compare(self, stats="elpd", min_model_size=0, round_to=None):
        """Return a DataFrame with the performance stats of the reference and submodels.

        Parameters:
        -----------
        stats : str
            The statistics to compute. Defaults to "elpd".
            * "elpd": expected log (pointwise) predictive density (ELPD).
            * "mlpd": mean log predictive density (MLPD), that is, the ELPD divided by the
            number of observations.
            * "gmpd": geometric mean predictive density (GMPD), that is, exp(MLDP).
            For discrete response families the GMPD is bounded by zero and one.
        min_model_size : int
            The minimum size of the submodels to compare. Defaults to 0, which means the
            intercept-only model is included in the comparison.
        round_to : int
            Number of decimals used to round results. Defaults to None

        Returns:
        --------
        DataFrame
            A DataFrame with the ELPD and standard error of the submodels and the reference model.
            The index of the DataFrame is the term names of the submodels, and the first row is the
            reference model.
        """
        # test that search has been previously run
        if not self._list_of_submodels:
            raise UserWarning("Please run search before comparing submodels.")

        if stats not in ["elpd", "mlpd", "gmpd"]:
            raise ValueError(
                "Please select one of the following statistics: 'elpd', 'mlpd', or 'gmpd'."
            )

        label_terms = []
        performance_info = {stats: [], "se": []}
        for k, submodel in enumerate(self._list_of_submodels):
            if k >= min_model_size:
                performance_info[stats].append(submodel.elpd)
                performance_info["se"].append(submodel.elpd_se)
                if submodel.term_names:
                    label_terms.append(submodel.term_names[-1])
                else:
                    label_terms.append("Intercept")

        label_terms.append("reference")
        performance_info[stats].append(self.reference_model.elpd)
        performance_info["se"].append(self.reference_model.elpd_se)

        if stats in ["mlpd", "gmpd"]:
            performance_info[stats] = np.array(performance_info[stats])
            performance_info["se"] = np.array(performance_info["se"])

            performance_info[stats] = performance_info[stats] / self._observed_array.shape[0]
            performance_info["se"] = performance_info["se"] / self._observed_array.shape[0]

            if stats == "gmpd":
                performance_info[stats] = np.exp(performance_info[stats])
                # delta method
                performance_info["se"] = performance_info["se"] * performance_info[stats]

        summary_df = DataFrame(performance_info, index=label_terms).iloc[::-1]

        if (round_to is not None) and (round_to not in ("None", "none")):
            summary_df = summary_df.round(round_to)

        return summary_df


def _get_base_terms(has_intercept, priors):
    """Extend the model term names to include dispersion terms."""

    base_terms = []
    # add intercept term if present
    if has_intercept:
        base_terms.append("Intercept")

    # add the auxiliary parameters
    if priors:
        aux_params = [f"{str(k)}" for k in priors]
        base_terms += aux_params
    return base_terms


def _check_interactions(term_names, method, require_lower_terms):
    """Check that interaction terms are not included without their main effects."""
    if method == "forward":
        interaction_terms = [term for term in term_names if ":" in term]
        if interaction_terms and require_lower_terms:
            missing_lower_terms = set()
            for interaction in interaction_terms:
                missing = _missing_lower_order_terms(interaction, term_names)
                missing_lower_terms.update(missing)
            if missing_lower_terms:
                raise ValueError(
                    "Interaction terms detected in the model, but the following lower-order "
                    f"terms are missing: {sorted(missing_lower_terms)}.\n"
                    "Please ensure that all lower-order interactions and main effects are included "
                    "in the model.\nIf you are sure that you want to exclude them, set "
                    "require_lower_terms=False to disable this check."
                )


class SubModel:
    """Submodel dataclass.

    Attributes:
        model (bambi.Model): The submodel's associated Bambi model, from which we can
            extract a built pymc model.
        idata (InferenceData): The inference data object of the submodel containing the
            projected posterior draws and log-likelihood.
        loo (float): The optimization loss of the submodel
        size (int): The number of common terms in the model, not including the intercept
        elpd (float): The expected log pointwise predictive density of the submodel
        elpd_se (float): The standard error of the expected log pointwise predictive
        term_names (list): The names of the terms in the model, including the intercept
        has_intercept (bool): Whether the model has an intercept term
    """

    def __init__(self, model, idata, loss, size, elpd, elpd_se, term_names, has_intercept):
        self.model = model
        self.idata = idata
        self.loss = loss
        self.size = size
        self.elpd = elpd
        self.elpd_se = elpd_se
        self.term_names = term_names
        self.has_intercept = has_intercept

    def __repr__(self) -> str:
        """String representation of the submodel."""
        if self.has_intercept:
            intercept = ["Intercept"]
        else:
            intercept = []

        return f"{intercept + self.term_names}"


class RefModel:
    """Reference model dataclass.

    Attributes:
        model (bambi.Model): The reference Bambi model, from which we can
            extract a built pymc model.
        idata (InferenceData): The inference data object of the reference model containing the
            posterior draws and log-likelihood.
        elpd (float): The expected log pointwise predictive density of the reference model
        elpd_se (float): The standard error of the expected log pointwise predictive
        term_names (list): The names of the terms in the model, including the intercept
        has_intercept (bool): Whether the model has an intercept term
    """

    def __init__(self, model, idata, elpd, elpd_se, term_names):
        self.bambi_model = model
        self.idata = idata
        self.size = len(term_names)
        self.elpd = elpd
        self.elpd_se = elpd_se
        self.term_names = term_names

    def __repr__(self) -> str:
        """String representation of the submodel."""
        return f"{self.term_names}"
