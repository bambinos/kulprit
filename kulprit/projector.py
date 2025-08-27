# pylint: disable=undefined-loop-variable
# pylint: disable=too-many-instance-attributes
"""Core reference model class."""
import warnings
from copy import copy
import numpy as np
from pandas import DataFrame

from bambi import formula

from kulprit.projection.arviz_io import compute_loo, get_observed_data, get_pps
from kulprit.projection.pymc_io import (
    add_switches,
    compile_mllk,
    compute_llk,
    turn_off_terms,
    get_model_information,
)
from kulprit.projection.search_strategies import user_path, forward_search, l1_search
from kulprit.projection.solver import solve


class ProjectionPredictive:
    """
    Projection Predictive class from which we perform the model selection procedure.

    Parameters:
    ----------
    model : Bambi model
        The reference GLM model to project
    idata : InferenceData
        The ArviZ InferenceData object of the fitted reference model
    rng : RandomState
        Random number generator used for sampling from the posterior if idata is not provided.
        And for sampling from the posterior predictive distribution.

    """

    def __init__(self, model, idata=None, rng=456):
        """Builder for projection predictive model selection."""
        # initialize attributes
        self.num_samples = None
        self.num_clusters = None
        self.early_stop = None
        self.tolerance = None
        self.pps = None
        self.ppc = None
        self.weights = None
        self.list_of_submodels = []

        # set random number generator
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(rng)

        # get information from Bambi's reference model
        self.model = model
        self.has_intercept = formula.formula_has_intercept(self.model.formula.main)
        self.response_name = self.model.response_component.term.name
        self.ref_family = self.model.family.name
        self.priors = self.model.constant_components
        self.ref_terms = [
            v.alias if v.alias is not None else k
            for k, v in model.components[model.family.likelihood.parent].common_terms.items()
        ]
        self.categorical_terms = sum(
            term.categorical
            for term in self.model.components[
                self.model.family.likelihood.parent
            ].common_terms.values()
        )

        self.base_terms = self._get_base_terms()

        # get information from PyMC's reference model
        if not self.model.built:
            self.model.build()
        self.pymc_model = copy(self.model.backend.model)
        initial_point = self.pymc_model.initial_point()
        self.ref_var_info = get_model_information(self.pymc_model, initial_point)
        self.all_terms = [fvar.name for fvar in self.pymc_model.free_RVs]
        self.initial_guess = np.concatenate([np.ravel(value) for value in initial_point.values()])

        # add switches to the model to turn on/off terms in the model
        # without having to rebuild the model
        self.pymc_model_sw, self.switches = add_switches(self.pymc_model, self.ref_terms)
        self.neg_log_likelihood = compile_mllk(self.pymc_model_sw, initial_point)

        # get information from ArviZ's InferenceData object
        self.idata = idata
        self._check_idata()
        self.observed_dataset, self.observed_array = get_observed_data(
            self.idata, self.response_name
        )

        self.elpd_ref = compute_loo(idata=self.idata)

    def __repr__(self) -> str:
        """Return the terms of the submodels."""

        if not self.list_of_submodels:
            return "ReferenceModel"

        else:
            str_of_submodels = "\n".join(
                f"{idx:>3} " f"{value.term_names}"
                for idx, value in enumerate(self.list_of_submodels)
            )
            return str_of_submodels

    def project(
        self,
        method="forward",
        user_terms=None,
        num_samples=400,
        num_clusters=20,
        early_stop=None,
        tolerance=1,
    ):
        """Perform model projection.

        Parameters:
        -----------
        method : str
            The search method to employ, either "forward" for a forward search, or "l1" for
            a L1-regularized search. Ignored if "terms" is provided.
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
        tolerance : float
            The tolerance for the optimization procedure. Defaults to 1. Decreasing this value
            will increase the accuracy of the projection at the cost of speed.
        """
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.early_stop = early_stop
        self.tolerance = tolerance
        self.pps, self.ppc, self.weights = get_pps(
            self.idata, self.response_name, self.num_samples, self.num_clusters, self.rng
        )

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
                if not set(term_names).issubset(self.all_terms):
                    raise ValueError(f"Term {idx} is not a valid term in the reference")

            self.list_of_submodels = user_path(self._project, user_terms)
        else:
            if method not in ["forward", "l1"]:
                raise ValueError("Please select either forward search or L1 search.")

            max_terms = len(self.model.components[self.model.family.likelihood.parent].common_terms)

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
                self.list_of_submodels = forward_search(
                    self._project, self.ref_terms, max_terms, self.elpd_ref, self.early_stop
                )
            else:
                # currently L1 search is not implemented for categorical models
                if self.categorical_terms:
                    raise NotImplementedError("Group-lasso not yet implemented")

                self.list_of_submodels = l1_search(
                    self._project,
                    self.model,
                    self.ref_terms,
                    max_terms,
                    self.elpd_ref,
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

        for submodel in self.list_of_submodels:
            if criterion == "mean":
                if (self.elpd_ref.elpd - submodel.elpd) < 4:
                    return submodel
            else:
                if submodel.elpd + submodel.elpd_se >= self.elpd_ref.elpd:
                    return submodel

        return None

    def _project(self, term_names, clusters=True):
        turn_off_terms(self.switches, self.ref_terms, term_names)

        if clusters:
            samples = self.ppc
            weights = self.weights
        else:
            samples = self.pps
            weights = None

        new_idata, loss = solve(
            self.neg_log_likelihood,
            samples,
            self.initial_guess,
            self.ref_var_info,
            weights,
            self.tolerance,
        )

        # Add observed data and log-likelihood to the projected InferenceData object
        # We only do this for the selected projected model, not the intermediate ones
        if new_idata is not None:
            new_idata.add_groups(observed_data=self.observed_dataset)
            new_idata.add_groups(log_likelihood=compute_llk(new_idata, self.pymc_model))
            # remove the variables that are not in the submodel
            vars_to_drop = [
                var
                for var in new_idata.posterior.data_vars
                if var not in (term_names + self.base_terms)
            ]
            new_idata.posterior = new_idata.posterior.drop_vars(vars_to_drop)

        # build SubModel object and return
        sub_model = SubModel(
            model=self.pymc_model_sw,
            idata=new_idata,
            loss=loss,
            elpd=None,
            elpd_se=None,
            size=len(term_names),
            term_names=term_names,
            has_intercept=self.has_intercept,
        )
        return sub_model

    def _get_base_terms(self):
        """Extend the model term names to include dispersion terms."""

        base_terms = []
        # add intercept term if present
        if self.has_intercept:
            base_terms.append("Intercept")

        # add the auxiliary parameters
        if self.priors:
            aux_params = [f"{str(k)}" for k in self.priors]
            base_terms += aux_params
        return base_terms

    def _check_idata(self):
        # build posterior if not provided
        if self.idata is None:
            warnings.warn("No InferenceData object provided. Building posterior from model.")
            self.idata = self.model.fit(
                idata_kwargs={"log_likelihood": True},
                random_seed=self.rng,
            )

        # check compatibility between model and idata
        if (
            not self.model.response_component.term.name
            == list(self.idata.observed_data.data_vars.variables)[0]
        ):
            raise UserWarning("Incompatible model and inference data.")

        # check if we have the log_likelihood group
        if "log_likelihood" not in self.idata.groups():
            warnings.warn(
                "log_likelihood group is missing from idata, it will be computed.\n"
                "To avoid this message, please run Bambi's fit method with the option "
                "idata_kwargs={'log_likelihood': True}"
            )
            self.model.compute_log_likelihood(self.idata)

        # check if we have the posterior_predictive group
        if "posterior_predictive" not in self.idata.groups():
            self.model.predict(self.idata, kind="response", inplace=True, random_seed=self.rng)

    def submodels(self, index):
        """Return submodels by index

        Parameters
        ----------
        index : int or list of int
            The index or indices of the submodels to return. If a list of indices is provided,
            the submodels will be returned in the order of the list.

        Returns
        -------
        SubModel(s)
            The submodel or list of submodels corresponding to the provided indices
        """

        if isinstance(index, int):
            return self.list_of_submodels[index]
        else:
            n_submodels = len(self.list_of_submodels)
            if not all(-n_submodels <= i < n_submodels for i in index):
                warnings.warn(
                    "At least one index is out of bounds. Ignoring out of bounds indices."
                )
                index = [i for i in index if -n_submodels <= i < n_submodels]
            return [self.list_of_submodels[i] for i in index]

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
        if not self.list_of_submodels:
            raise UserWarning("Please run search before comparing submodels.")

        if stats not in ["elpd", "mlpd", "gmpd"]:
            raise ValueError(
                "Please select one of the following statistics: 'elpd', 'mlpd', or 'gmpd'."
            )

        label_terms = []
        performance_info = {stats: [], "se": []}
        for k, submodel in enumerate(self.list_of_submodels):
            if k >= min_model_size:
                performance_info[stats].append(submodel.elpd)
                performance_info["se"].append(submodel.elpd_se)
                if submodel.term_names:
                    label_terms.append(submodel.term_names[-1])
                else:
                    label_terms.append("Intercept")

        label_terms.append("reference")
        performance_info[stats].append(self.elpd_ref.elpd)
        performance_info["se"].append(self.elpd_ref.se)

        if stats in ["mlpd", "gmpd"]:
            performance_info[stats] = np.array(performance_info[stats])
            performance_info["se"] = np.array(performance_info["se"])

            performance_info[stats] = performance_info[stats] / self.observed_array.shape[0]
            performance_info["se"] = performance_info["se"] / self.observed_array.shape[0]

            if stats == "gmpd":
                performance_info[stats] = np.exp(performance_info[stats])
                # delta method
                performance_info["se"] = performance_info["se"] * performance_info[stats]

        summary_df = DataFrame(performance_info, index=label_terms).iloc[::-1]

        if (round_to is not None) and (round_to not in ("None", "none")):
            summary_df = summary_df.round(round_to)

        return summary_df


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
