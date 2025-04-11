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
        self.ref_var_info = get_model_information(self.pymc_model)
        self.all_terms = [fvar.name for fvar in self.pymc_model.free_RVs]

        # get information from ArviZ's InferenceData object
        self.idata = idata
        self._check_idata()
        self.observed_dataset, self.observed_array = get_observed_data(
            self.idata, self.response_name
        )
        self.num_samples = None
        # self.idata = compute_pps(self.pymc_model, self.idata)
        self.elpd_ref = compute_loo(idata=self.idata)

        self.tolerance = None
        self.early_stop = None
        self.pps = None
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
        self, max_terms=None, path="forward", num_samples=100, tolerance=0.01, early_stop=False
    ):
        """Perform model projection.

        If ``max_terms`` is not provided, then the search path runs from the intercept-only model
        up to but not including the full model.

        Parameters:
        -----------
        max_terms : int
            The number of parameters of the largest submodel in the search path, not including the
            intercept term.
        path : str or list
            The search method to employ, either "forward" for a forward search, or "l1" for
            a L1-regularized search path. If a nested list of terms is provided, model with
            those terms will be projected directly.
        num_samples : int
            The number of samples to draw from the posterior predictive distribution for the
            projection procedure. Defaults to 100.
        tolerance : float
            The tolerance for the optimization procedure. Defaults to 0.01
        early_stop : bool or str
            Whether to stop the search when the difference in ELPD between the submodel and the
            reference model is small. There are two criteria, "mean" and "se". The "mean" criterion
            stops the search when the difference between a the ELPD is smaller than 4. The "se"
            criterion stops the search when the ELPD of the submodel is within one standard error
            of the reference model. Defaults to False.
        """
        self.num_samples = num_samples
        self.tolerance = tolerance
        self.early_stop = early_stop
        self.pps = get_pps(self.idata, self.response_name, self.num_samples)

        # test if path is a list of terms
        if isinstance(path, list):
            # check if the length of the path always increase
            if not all(len(path[idx]) < len(path[idx + 1]) for idx in range(len(path) - 1)):
                raise ValueError("Please provide a list of terms in increasing order")
            # test if the terms in the path are valid
            for idx, term_names in enumerate(path):
                if not set(term_names).issubset(self.all_terms):
                    raise ValueError(f"Term {idx} is not a valid term in the reference")

            self.list_of_submodels = user_path(self._project, path)
        else:
            # test valid solution method
            if path not in ["forward", "l1"]:
                raise ValueError("Please select either forward search or L1 search.")

            # set default `max_terms` value
            n_terms = len(self.model.components[self.model.family.likelihood.parent].common_terms)
            if max_terms is None:
                max_terms = n_terms
            # test `max_terms` input
            elif max_terms > n_terms:
                warnings.warn(
                    "max_terms is larger than the number of terms in the reference model."
                    + "Searching for {n_terms}."
                )
                max_terms = n_terms

            if path == "forward":
                self.list_of_submodels = forward_search(
                    self._project, self.ref_terms, max_terms, self.elpd_ref, self.early_stop
                )
            else:
                # test whether the model includes categorical terms, and if so raise error
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

    def _project(self, term_names):

        term_names_ = self.base_terms + term_names
        new_model = compute_new_model(
            self.pymc_model, self.ref_var_info, self.all_terms, term_names_
        )

        neg_log_likelihood, old_y_value, obs_rvs = compile_mllk(new_model)
        initial_guess = np.concatenate(
            [np.ravel(value) for value in new_model.initial_point().values()]
        )
        var_info = get_model_information(new_model)

        new_idata, loss = solve(
            neg_log_likelihood,
            self.pps,
            initial_guess,
            var_info,
            self.tolerance,
        )
        # restore obs_rvs value in the model
        new_model.rvs_to_values[obs_rvs] = old_y_value

        # Add observed data and log-likelihood to the projected InferenceData object
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
            with self.pymc_model:
                self.idata = sample(idata_kwargs={"log_likelihood": True}, random_seed=self.rng)
                sample_posterior_predictive(
                    self.idata, extend_inferencedata=True, random_seed=self.rng
                )

        if "log_likelihood" not in self.idata.groups():
            raise UserWarning(
                """Please run Bambi's fit method with the option
                idata_kwargs={'log_likelihood': True}"""
            )
        if "posterior_predictive" not in self.idata.groups():
            self.model.predict(self.idata, kind="response", inplace=True)

        # test compatibility between model and idata
        if (
            not self.model.response_component.term.name
            == list(self.idata.observed_data.data_vars.variables)[0]
        ):
            raise UserWarning("Incompatible model and inference data.")

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
        include_reference=True,
        labels="formula",
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
