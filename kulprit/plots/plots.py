from arviz.plots.plot_utils import _scale_fig_size
import matplotlib.pyplot as plt
import numpy as np


def plot_compare(cmp_df, legend=True, title=True, figsize=None, plot_kwargs=None):
    """
    Plot model comparison.

    Parameters
    ----------
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

    fig, ax1 = plt.subplots(1, figsize=figsize)

    # double axes
    ax2 = ax1.twinx()

    ax1.errorbar(
        y=cmp_df["elpd_loo"][1:],
        x=xticks_pos[::2],
        yerr=cmp_df.se[1:],
        label="Submodel ELPD",
        color=plot_kwargs.get("color_eldp", "k"),
        fmt=plot_kwargs.get("marker_eldp", "o"),
        mfc=plot_kwargs.get("marker_fc_elpd", "white"),
        mew=linewidth,
        lw=linewidth,
        markersize=4,
    )
    ax2.errorbar(
        y=cmp_df["elpd_diff"].iloc[1:],
        x=xticks_pos[1::2],
        yerr=cmp_df.dse[1:],
        label="ELPD difference\n(to reference model)",
        color=plot_kwargs.get("color_dse", "grey"),
        fmt=plot_kwargs.get("marker_dse", "^"),
        mew=linewidth,
        elinewidth=linewidth,
        markersize=4,
    )

    ax1.axhline(
        cmp_df["elpd_loo"].iloc[0],
        ls=plot_kwargs.get("ls_reference", "--"),
        color=plot_kwargs.get("color_ls_reference", "grey"),
        lw=linewidth,
        label="Reference model ELPD",
    )

    if legend:
        fig.legend(
            bbox_to_anchor=(0.9, 0.1),
            loc="lower right",
            ncol=1,
            fontsize=ax_labelsize * 0.6,
        )

    if title:
        ax1.set_title(
            "Model comparison",
            fontsize=ax_labelsize * 0.6,
        )

    # remove double ticks
    xticks_pos, xticks_labels = xticks_pos[::2], xticks_labels[::2]

    # set axes
    ax1.set_xticks(xticks_pos)
    ax1.set_ylabel("ELPD", fontsize=ax_labelsize * 0.6)
    ax1.set_xlabel("Submodel size", fontsize=ax_labelsize * 0.6)
    ax1.set_xticklabels(xticks_labels)
    ax1.set_xlim(-1 + step, 0 - step)
    ax1.tick_params(labelsize=xt_labelsize * 0.6)
    ax2.set_ylabel("ELPD difference", fontsize=ax_labelsize * 0.6, color="grey")
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.tick_params(axis="y", colors="grey")
    align_yaxis(ax1, cmp_df["elpd_loo"].iloc[0], ax2, 0)

    return ax1


def align_yaxis(ax1, v_1, ax2, v_2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y_1 = ax1.transData.transform((0, v_1))
    _, y_2 = ax2.transData.transform((0, v_2))
    inv = ax2.transData.inverted()
    _, d_y = inv.transform((0, 0)) - inv.transform((0, y_1 - y_2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + d_y, maxy + d_y)
