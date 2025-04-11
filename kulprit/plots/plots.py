from arviz.plots.plot_utils import _scale_fig_size
from arviz.plots import plot_density, plot_forest
import matplotlib.pyplot as plt
import numpy as np


def plot_compare(
    elpd_info, label_terms=None, legend=True, title=True, figsize=None, plot_kwargs=None
):
    """
    Plot model comparison.

    Parameters:
    -----------
    cmp_df : pd.DataFrame
        Dataframe containing the comparison data. Should have columns
        `elpd_loo` and `elpd_diff` containing the ELPD values and the
        differences to the reference model.
    label_terms : list
        List of the labels for the submodels.
    legend : bool
        Flag for plotting the legend, default True.
    title : bool
        Flag for plotting the title, default True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    plot_kwargs : dict
        Dictionary of plot parameters. Available keys:
        - color_eldp : color for the ELPD points
        - marker_eldp : marker for the ELPD points
        - marker_fc_elpd : face color for the ELPD points
        - ls_reference : linestyle for the reference model line
        - color_ls_reference : color for the reference model line
        - xlabel_rotation : rotation for the x-axis labels

    """
    if plot_kwargs is None:
        plot_kwargs = {}

    if figsize is None:
        figsize = (10, 4)

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, None, 1, 1)

    xticks_pos = np.linspace(0, 1, len(elpd_info) - 1)
    xticks_num_labels = [value[0] for value in elpd_info[1:]]
    xticks_name_labels = [f"\n\n{term}" for term in label_terms]
    elpd_loo = [value[1] for value in elpd_info]
    elpd_se = [value[2] for value in elpd_info]

    fig, axes = plt.subplots(1, figsize=figsize)

    axes.errorbar(
        y=elpd_loo[1:],
        x=xticks_pos,
        yerr=elpd_se[1:],
        label="Submodels",
        color=plot_kwargs.get("color_eldp", "k"),
        fmt=plot_kwargs.get("marker_eldp", "o"),
        mfc=plot_kwargs.get("marker_fc_elpd", "white"),
        mew=linewidth,
        lw=linewidth,
        markersize=4,
    )

    axes.axhline(
        elpd_loo[0],
        ls=plot_kwargs.get("ls_reference", "--"),
        color=plot_kwargs.get("color_ls_reference", "grey"),
        lw=linewidth,
        label="Reference model",
    )

    axes.fill_between(
        [-0.15, 1.15],
        elpd_loo[0] + elpd_se[0],
        elpd_loo[0] - elpd_se[0],
        alpha=0.1,
        color=plot_kwargs.get("color_ls_reference", "grey"),
    )

    if legend:
        fig.legend(
            bbox_to_anchor=(0.9, 0.3),
            loc="lower right",
            ncol=1,
            fontsize=ax_labelsize * 0.6,
        )

    if title:
        axes.set_title(
            "Model comparison",
            fontsize=ax_labelsize * 0.6,
        )

    sec0 = axes.secondary_xaxis(location=0)
    sec0.set_xticks(xticks_pos, xticks_num_labels)
    sec0.tick_params("x", length=0, labelsize=xt_labelsize * 0.6)

    sec1 = axes.secondary_xaxis(location=0)
    sec1.set_xticks(xticks_pos, xticks_name_labels, rotation=plot_kwargs.get("xlabel_rotation", 0))
    sec1.tick_params("x", length=0, labelsize=xt_labelsize * 0.6)
    sec1.set_xlabel("Submodels", fontsize=ax_labelsize * 0.6)

    axes.set_xticks([])
    axes.set_ylabel("ELPD", fontsize=ax_labelsize * 0.6)
    axes.set_xlim(-0.1, 1.1)
    axes.tick_params(labelsize=xt_labelsize * 0.6)

    return axes


def plot_densities(
    model,
    idata,
    submodels,
    var_names=None,
    include_reference=True,
    labels="size",
    kind="density",
    figsize=None,
    plot_kwargs=None,
):
    """Compare the projected posterior densities of the submodels"""
    if plot_kwargs is None:
        plot_kwargs = {}

    if kind not in ["density", "forest"]:
        raise ValueError("kind must be one of 'density' or 'forest'")

    # set default variable names to the reference model terms
    if not var_names:
        if include_reference:
            var_names = [fvar.name for fvar in model.backend.model.free_RVs]
        else:
            var_names = ["Intercept"] + sorted(submodel.term_names for submodel in submodels)[-1]

    if include_reference:
        data = [idata]
        l_labels = ["Reference"]
    else:
        data = []
        l_labels = []

    if labels == "formula":
        l_labels.extend([",".join(submodel.term_names) for submodel in submodels])
    else:
        l_labels.extend([submodel.size for submodel in submodels])

    data.extend([submodel.idata for submodel in submodels])

    if kind == "density":
        plot_kwargs.setdefault("outline", False)
        plot_kwargs.setdefault("shade", 0.4)
        plot_kwargs.setdefault("figsize", figsize)

        axes = plot_density(
            data=data,
            var_names=var_names,
            data_labels=l_labels,
            **plot_kwargs,
        )

    else:
        plot_kwargs.setdefault("combined", True)
        plot_kwargs.setdefault("figsize", (10, 2 + len(var_names) ** 0.5))

        axes = plot_forest(
            data=data,
            model_names=l_labels,
            var_names=var_names,
            **plot_kwargs,
        )

    return axes
