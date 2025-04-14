# pylint: disable=undefined-loop-variable
# pylint: disable=too-many-instance-attributes
"""Core reference model class."""
import warnings
from copy import copy
import numpy as np

from bambi import formula
from pymc import sample, sample_posterior_predictive
from kulprit.plots.plots import plot_compare, plot_densities

from kulprit.projection.arviz_io import compute_loo, get_observed_data, get_pps
from kulprit.projection.pymc_io import (
    compile_mllk,
    compute_llk,
    compute_new_model,
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
        self.ref_terms = list(
            self.model.components[self.model.family.likelihood.parent].common_terms.keys()
        )
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
        self.ref_var_info = get_model_information(self.pymc_model, self.pymc_model.initial_point())
        self.all_terms = [fvar.name for fvar in self.pymc_model.free_RVs]

        # get information from ArviZ's InferenceData object
        self.idata = idata
        self._check_idata()
        self.observed_dataset, self.observed_array = get_observed_data(
            self.idata, self.response_name
        )
        self.num_samples = None
        self.num_clusters = None
        self.elpd_ref = compute_loo(idata=self.idata)

        self.tolerance = None
        self.early_stop = None
        self.pps = None
        self.ppc = None
        self.weights = None
        self.list_of_submodels = []

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
        tolerance=0.01,
        num_clusters=20,
        early_stop=None,
    ):
        """Perform model projection.

        Parameters:
        -----------
        method : str or list
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
        tolerance : float
            The tolerance for the optimization procedure. Defaults to 0.01
        early_stop : str or int, optional
            Whether to stop the search earlier. If an integer is provided, the search stops
            when the submodel size is equal to the integer. If a string is provided, the search
            stops when the difference in ELPD between the reference and submodel is small.
            There are two criteria to define what small is, "mean" and "se".
            The "mean" criterion stops the search when the difference between a the ELPD is smaller
            than 4. The "se" criterion stops the search when the ELPD of the submodel is within
            one standard error of the reference model. Defaults to None.
        """
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.early_stop = early_stop
        self.pps, self.ppc, self.weights = get_pps(
            self.idata, self.response_name, self.num_samples, self.num_clusters, self.rng
        )

        # if user provided the terms we used them directly, no search is performed
        if user_terms is not None:
            # check if the terms are a list of lists
            if not isinstance(user_terms, list) or not all(isinstance(term, list ) for term in user_terms):
                raise ValueError("Please provide a list of lists of terms.")
            # check if the length of the submodels always increase
            if not all(len(user_terms[idx]) < len(user_terms[idx + 1]) for idx in range(len(user_terms) - 1)):
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

            # if early_stop is an integer, check that it is positive and not larger than the number of terms
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
                if (self.elpd_ref.elpd_loo - submodel.elpd_loo) < 4:
                    return submodel
            else:
                if submodel.elpd_loo + submodel.elpd_se >= self.elpd_ref.elpd_loo:
                    return submodel

        return None

    def _project(self, term_names, clusters=True):
        term_names_ = self.base_terms + term_names
        new_model = compute_new_model(
            self.pymc_model, self.ref_var_info, self.all_terms, term_names_
        )

        initial_point = new_model.initial_point()
        neg_log_likelihood, old_y_value, obs_rvs, initial_point = compile_mllk(
            new_model, initial_point
        )

        initial_guess = np.concatenate([np.ravel(value) for value in initial_point.values()])
        var_info = get_model_information(new_model, initial_point)

        if clusters:
            samples = self.ppc
            weights = self.weights
        else:
            samples = self.pps
            weights = None

        new_idata, loss = solve(
            neg_log_likelihood,
            samples,
            initial_guess,
            var_info,
            self.tolerance,
            weights,
        )
        
        # restore obs_rvs value in the model
        new_model.rvs_to_values[obs_rvs] = old_y_value

        # Add observed data and log-likelihood to the projected InferenceData object
        # We only do this for the selected projected model, not the intermediate ones
        if new_idata is not None:
            new_idata.add_groups(observed_data=self.observed_dataset)
            new_idata.add_groups(log_likelihood=compute_llk(new_idata, new_model))

        # build SubModel object and return
        sub_model = SubModel(
            model=new_model,
            idata=new_idata,
            loss=loss,
            elpd_loo=None,
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
            warnings.warn("log_likelihood group is missing from idata, it will be computed.\n"
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

    def compare(
        self,
        plot=True,
        min_model_size=0,
        legend=True,
        title=True,
        figsize=None,
        plot_kwargs=None,
    ):
        """Compare the ELPD of the projected models along the search path.


        Parameters:
        -----------
        plot : bool
            Plot the results of the comparison. Defaults to True
        legend : bool
            Add legend to figure. Defaults to True.
        title : bool
            Show a tittle with a description of how to interpret the plot. Defaults to True.
        figsize : tuple
            If None, size is (10, num of submodels) inches
        plot_kwargs : dict
            Optional arguments for plot elements. Currently accepts 'color_elpd', 'marker_elpd',
        'marker_fc_elpd', 'color_dse', 'marker_dse', 'ls_reference', 'color_ls_reference',
        'xlabel_rotation'.

        Returns:
        --------
        cmp : elpd_info
            tuples of index, elpd_loo point estimate and standard error for each submodel
            The index -1 corresponds to the reference model.
        axes : matplotlib_axes
        """
        # test that search has been previously run
        if not self.list_of_submodels:
            raise UserWarning("Please run search before comparing submodels.")

        # initiate plotting arguments if none provided
        if plot_kwargs is None:
            plot_kwargs = {}

        label_terms = []
        elpd_info = [(-1, self.elpd_ref.elpd_loo, self.elpd_ref.se)]
        # make list with elpd loo and se for each submodel
        for k, submodel in enumerate(self.list_of_submodels):
            if k >= min_model_size:
                elpd_info.append((k, submodel.elpd_loo, submodel.elpd_se))
                if submodel.term_names:
                    label_terms.append(submodel.term_names[-1])
                else:
                    label_terms.append("Intercept")

        # plot the comparison if requested
        axes = None
        if plot:
            axes = plot_compare(elpd_info, label_terms, legend, title, figsize, plot_kwargs)

        return elpd_info, axes

    def plot_densities(
        self,
        var_names=None,
        submodels=None,
        include_reference=False,
        labels="size",
        kind="density",
        figsize=None,
        plot_kwargs=None,
    ):
        """Compare the projected posterior densities of the submodels

        Parameters:
        -----------
        var_names : list of str, optional
            List of variables to plot.
        submodels : list of int, optional
            List of submodels to plot, 0 is intercept-only model and the largest valid integer is
            the total number of variables in reference model. If None, all submodels are plotted.
        include_reference : bool
            Whether to include the reference model in the plot. Defaults to True.
        labels : str
            If "formula", the labels are the formulas of the submodels. If "size", the number
            of covariates in the submodels.
        kind : str
            The kind of plot to create. Either "density" or "forest". Defaults to "density".
        figsize : tuple
            Figure size. If None it will be defined automatically.
        plot_kwargs : dict
            Dictionary passed to ArviZ's ``plot_density`` function (if kind density) or to
            ``plot_forest`` (if kind forest).

        Returns:
        --------

        axes : matplotlib_axes
        """
        if submodels is None:
            submodels = self.list_of_submodels
        else:
            submodels = self.submodels(submodels)

        return plot_densities(
            self.model,
            self.idata,
            var_names=var_names,
            submodels=submodels,
            include_reference=include_reference,
            labels=labels,
            kind=kind,
            figsize=figsize,
            plot_kwargs=plot_kwargs,
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
        elpd_loo (float): The expected log pointwise predictive density of the submodel
        elpd_se (float): The standard error of the expected log pointwise predictive
        term_names (list): The names of the terms in the model, including the intercept
        has_intercept (bool): Whether the model has an intercept term
    """

    def __init__(self, model, idata, loss, size, elpd_loo, elpd_se, term_names, has_intercept):
        self.model = model
        self.idata = idata
        self.loss = loss
        self.size = size
        self.elpd_loo = elpd_loo
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
