from arviz_plots import plot_dist as azp_plot_dist
from arviz_plots import plot_forest as azp_plot_forest
from arviz_plots import plot_compare as azp_plot_compare


def plot_compare(
    cmp_df,
    relative_scale=True,
    backend=None,
    visuals=None,
    **pc_kwargs,
):
    r"""Summary plot for model comparison.

    Models are compared based on their expected log pointwise predictive density (ELPD).
    Higher ELPD values indicate better predictive performance.

    The ELPD is estimated by Pareto smoothed importance sampling leave-one-out
    cross-validation (LOO). Details are presented in [1]_ and [2]_.

    The ELPD can only be interpreted in relative terms. But differences in ELPD less than 4
    are considered negligible [3]_.

    Parameters
    ----------
    comp_df : pandas.DataFrame
        The result of Kulprit's `compare` function
    relative_scale : bool, optional.
        If True scale the ELPD values relative to the reference model.
        Defaults to True.
    backend : {"bokeh", "matplotlib", "plotly"}
        Select plotting backend. Defaults to ArviZ's rcParams["plot.backend"].
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * point_estimate -> passed to :func:`~arviz_plots.backend.none.scatter`
        * error_bar -> passed to :func:`~arviz_plots.backend.none.line`
        * ref_line -> passed to :func:`~arviz_plots.backend.none.hline`.
        * ref_band -> passed to :func:`~arviz_plots.backend.none.hspan`
        * similar_line -> passed to :func:`~arviz_plots.backend.none.hline` or
          Defaults to False
        * labels -> passed to :func:`~arviz_plots.backend.none.xticks`
          and :func:`~arviz_plots.backend.none.yticks`
        * title -> passed to :func:`~arviz_plots.backend.none.title`.
          Defaults to False.
        * ticklabels -> passed to :func:`~arviz_plots.backend.none.yticks`

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    References
    ----------
    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017).
        https://doi.org/10.1007/s11222-016-9696-4. arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646

    .. [3] Sivula et al. *Uncertainty in Bayesian Leave-One-Out Cross-Validation Based Model
        Comparison*. (2025). https://doi.org/10.48550/arXiv.2008.10296
    """
    if visuals is None:
        visuals = {}

    visuals.setdefault("title", False)
    visuals.setdefault("ref_band", True)
    visuals.setdefault("similar_line", False)

    pc = azp_plot_compare(
        cmp_df,
        relative_scale=relative_scale,
        hide_top_model=True,
        rotated=True,
        visuals=visuals,
        backend=backend,
        **pc_kwargs,
    )
    return pc


def plot_forest(
    ppi,
    submodels=None,
    include_reference=False,
    var_names=None,
    filter_vars=None,
    coords=None,
    sample_dims=None,
    point_estimate=None,
    ci_kind=None,
    ci_probs=None,
    labels=None,
    shade_label=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals=None,
    visuals=None,
    stats=None,
    **pc_kwargs,
):
    """Plot 1D marginal credible intervals in a single plot.

    This function is a thin wrapper around :func:`arviz.plots.plot_forest`
    that prepares the data from a :class:`kulprit.ProjectionPredictive` object.

    Parameters
    ----------
    ppi :`kulprit.ProjectionPredictive` object
    submodels : list of {int, str}, optional
        List of submodel sizes or names to be plotted.
    include_reference : bool, default False
        Whether to include the reference model in the plot.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    combined : bool, default False
        Whether to plot intervals for each chain or not. Ignored when the "chain" dimension
        is not present.
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_probs : (float, float), optional
        Indicates the probabilities that should be contained within the plotted credible intervals.
        It should be sorted as the elements refer to the probabilities of the "trunk" and "twig"
        elements. Defaults to ``(0.5, rcParams["stats.ci_prob"])``
    labels : sequence of str, optional
        Sequence with the dimensions to be labelled in the plot. By default all dimensions
        except "chain" and "model" (if present). The order of `labels` is ignored,
        only elements being present in it matters.
        It can include the special "__variable__" indicator, and does so by default.
    shade_label : str, default None
        Element of `labels` that should be used to add shading horizontal strips to the plot.
        Note that labels and credible intervals are plotted in different :term:`plots`.
        The shading is applied to both plots, and the spacing between them is set to 0
        *if possible*, which is not always the case (one notable example being matplotlib's
        constrained layout).
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except "ticklabels"
        and "remove_axis" which do not apply, and "twig" and "trunk" which
        take the same aesthetics through the "credible_interval" key.

        By default, aesthetic mappings are generated for: y, alpha, overlay and color
        (if multiple models are present). All aesthetic mappings but alpha are applied
        to both the credible intervals and the point estimate; overlay is applied
        to labels; and both overlay and alpha are applied to the shade.

        "overlay" is a dummy aesthetic to trigger looping over variables and/or
        dimensions using all aesthetics in every iteration. "alpha" gets two
        values (0, 0.3) in order to trigger the alternate shading effect.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * trunk, twig -> passed to :func:`~.visuals.line_x`
        * point_estimate -> passed to :func:`~.visuals.scatter_x`
        * labels -> passed to :func:`~.visuals.annotate_label`
        * shade -> passed to :func:`~.visuals.fill_between_y`
        * ticklabels -> passed to :func:`~.backend.xticks`
        * remove_axis -> not passed anywhere, can only take ``False`` as value to skip calling
          :func:`~.visuals.remove_axis`

    stats : mapping, optional
        Valid keys are:

        * trunk, twig -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection
    """
    models_to_plot, var_names = _get_models_to_plot(ppi, var_names, submodels, include_reference)

    if stats is None:
        stats = {}

    stats.setdefault("trunk", {"skipna": True})
    stats.setdefault("twig", {"skipna": True})

    pc = azp_plot_forest(
        models_to_plot,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        sample_dims=sample_dims,
        combined=True,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_probs=ci_probs,
        labels=labels,
        shade_label=shade_label,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
        **pc_kwargs,
    )

    pc.add_legend("model")

    return pc


def plot_dist(
    ppi,
    submodels=None,
    include_reference=False,
    var_names=None,
    filter_vars=None,
    coords=None,
    sample_dims=None,
    kind=None,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals=None,
    visuals=None,
    stats=None,
    **pc_kwargs,
):
    """Plot 1D marginal densities.

    This function is a thin wrapper around :func:`arviz_plots.plot_dist`
    that prepares the data from a :class:`kulprit.ProjectionPredictive` object.

    Parameters
    ----------
    ppi :`kulprit.ProjectionPredictive` object
    submodels : list of {int, str}, optional
        List of submodel sizes or names to be plotted.
    include_reference : bool, default False
        Whether to include the reference model in the plot.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        With a single model, no aesthetic mappings are generated by default,
        each variable+coord combination gets a :term:`plot` but they all look the same,
        unless there are user provided aesthetic mappings.
        With multiple models, ``plot_dist`` maps "color" and "y" to the "model" dimension.

        By default, all aesthetics but "y" are mapped to the density representation,
        and if multiple models are present, "color" and "y" are mapped to the
        credible interval and the point estimate.

        When "point_estimate" key is provided but "point_estimate_text" isn't,
        the values assigned to the first are also used for the second.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.step_hist`

        * face -> :term:`visual` that fills the area under the marginal distribution representation.

          Defaults to False. Depending on the value of `kind` it is passed to:

          * "kde" or "ecdf" -> passed to :func:`~arviz_plots.visuals.fill_between_y`
          * "hist" -> passed to :func:`~arviz_plots.visuals.hist`

        * credible_interval -> passed to :func:`~arviz_plots.visuals.line_x`. Defaults to False.
        * point_estimate -> passed to :func:`~arviz_plots.visuals.scatter_x`. Defaults to False.
        * point_estimate_text -> passed to :func:`~arviz_plots.visuals.point_estimate_text`. False.
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * rug -> passed to :func:`~arviz_plots.visuals.scatter_x`. Defaults to False.
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats : mapping, optional
        Valid keys are:

        * dist -> passed to kde, ecdf, ...
        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection
    """

    if visuals is None:
        visuals = {}

    if stats is None:
        stats = {}

    visuals.setdefault("point_estimate_text", False)
    visuals.setdefault("credible_interval", False)
    stats.setdefault("point_estimate", {"skipna": True})

    models_to_plot, var_names = _get_models_to_plot(ppi, var_names, submodels, include_reference)

    pc = azp_plot_dist(
        models_to_plot,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
        **pc_kwargs,
    )

    pc.add_legend("model")
    return pc


def _get_models_to_plot(ppi, var_names, submodels, include_reference):
    """Prepare a dictionary of models to be plotted."""
    if submodels is None:
        submodels = ppi.list_of_submodels
    else:
        submodels = ppi.submodels(submodels)

    if not var_names:
        if include_reference:
            var_names = ["Intercept"] + ppi.ref_terms
        else:
            var_names = ["Intercept"] + sorted(submodel.term_names for submodel in submodels)[-1]

    if include_reference:
        models_to_plot = {"Reference": ppi.idata.posterior}
    else:
        models_to_plot = {}

    models_to_plot.update({submodel.size: submodel.idata.posterior for submodel in submodels})

    return models_to_plot, var_names
