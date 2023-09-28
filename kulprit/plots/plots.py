from arviz.plots.plot_utils import _scale_fig_size
from arviz.plots import plot_density, plot_forest
import matplotlib.pyplot as plt
import numpy as np


def plot_compare(cmp_df, legend=True, title=True, figsize=None, plot_kwargs=None):
    """
    Plot model comparison.

    Parameters:
    -----------
    cmp_df : pd.DataFrame
        Dataframe containing the comparison data. Should have columns
        `elpd_loo` and `elpd_diff` containing the ELPD values and the
        differences to the reference model.
    legend : bool
        Flag for plotting the legend, default True.
    title : bool
        Flag for plotting the title, default True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    plot_kwargs : dict
    """

    if plot_kwargs is None:
        plot_kwargs = {}

    if figsize is None:
        figsize = (len(cmp_df) - 1, 10)

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, None, 1, 1)

    xticks_pos, step = np.linspace(0, -1, ((cmp_df.shape[0]) * 2) - 2, retstep=True)
    xticks_pos[1::2] = xticks_pos[1::2] - step * 1.5

    labels = cmp_df.index.values[1:]
    xticks_labels = [""] * len(xticks_pos)
    xticks_labels[0] = labels[0]
    xticks_labels[2::2] = labels[1:]

    fig, axes = plt.subplots(1, figsize=figsize)

    axes.errorbar(
        y=cmp_df["elpd_loo"][1:],
        x=xticks_pos[::2],
        yerr=cmp_df.se[1:],
        label="Submodels",
        color=plot_kwargs.get("color_eldp", "k"),
        fmt=plot_kwargs.get("marker_eldp", "o"),
        mfc=plot_kwargs.get("marker_fc_elpd", "white"),
        mew=linewidth,
        lw=linewidth,
        markersize=4,
    )

    axes.axhline(
        cmp_df["elpd_loo"].iloc[0],
        ls=plot_kwargs.get("ls_reference", "--"),
        color=plot_kwargs.get("color_ls_reference", "grey"),
        lw=linewidth,
        label="Reference model",
    )

    axes.fill_between(
        [-2, 1],
        cmp_df["elpd_loo"].iloc[0] + cmp_df["se"].iloc[0],
        cmp_df["elpd_loo"].iloc[0] - cmp_df["se"].iloc[0],
        alpha=0.1,
        color=plot_kwargs.get("color_ls_reference", "grey"),
    )

    if legend:
        fig.legend(
            bbox_to_anchor=(0.9, 0.1),
            loc="lower right",
            ncol=1,
            fontsize=ax_labelsize * 0.6,
        )

    if title:
        axes.set_title(
            "Model comparison",
            fontsize=ax_labelsize * 0.6,
        )

    # remove double ticks
    xticks_pos, xticks_labels = xticks_pos[::2], xticks_labels[::2]

    # set axes
    axes.set_xticks(xticks_pos)
    axes.set_ylabel("ELPD", fontsize=ax_labelsize * 0.6)
    axes.set_xlabel("Submodel size", fontsize=ax_labelsize * 0.6)
    axes.set_xticklabels(xticks_labels)
    axes.set_xlim(-1 + step, 0 - step)
    axes.tick_params(labelsize=xt_labelsize * 0.6)

    return axes


def plot_densities(
    model,
    path,
    idata,
    var_names=None,
    submodels=None,
    include_reference=True,
    labels="formula",
    kind="density",
    figsize=None,
    plot_kwargs=None,
):
    """Compare the projected posterior densities of the submodels"""

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("figsize", figsize)

    if kind not in ["density", "forest"]:
        raise ValueError("kind must be one of 'density' or 'forest'")

    # set default variable names to the reference model terms
    if not var_names:
        var_names = list(set(model.response_component.terms.keys()) - set([model.response_name]))

    if include_reference:
        data = [idata]
        l_labels = ["Reference"]
        var_names.append(f"~{model.response_name}_mean")
    else:
        data = []
        l_labels = []

    if submodels is None:
        submodels = path.values()
    else:
        submodels = [path[key] for key in submodels]

    if labels == "formula":
        l_labels.extend([submodel.model.formula for submodel in submodels])
    else:
        l_labels.extend([submodel.size for submodel in submodels])

    data.extend([submodel.idata for submodel in submodels])

    if kind == "density":
        plot_kwargs.setdefault("outline", False)
        plot_kwargs.setdefault("shade", 0.4)

        axes = plot_density(
            data=data,
            var_names=var_names,
            data_labels=l_labels,
            **plot_kwargs,
        )

    elif kind == "forest":
        plot_kwargs.setdefault("combined", True)
        axes = plot_forest(
            data=data,
            model_names=l_labels,
            var_names=var_names,
            **plot_kwargs,
        )

    return axes


def align_yaxis(axes, v_1, ax2, v_2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in axes"""
    _, y_1 = axes.transData.transform((0, v_1))
    _, y_2 = ax2.transData.transform((0, v_2))
    inv = ax2.transData.inverted()
    _, d_y = inv.transform((0, 0)) - inv.transform((0, y_1 - y_2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + d_y, maxy + d_y)
