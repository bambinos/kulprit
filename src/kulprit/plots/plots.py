from arviz.plots.plot_utils import _scale_fig_size
import matplotlib.pyplot as plt
import numpy as np


def plot_compare(cmp_df, legend=True, title=True, figsize=None, plot_kwargs=None):

    if plot_kwargs is None:
        plot_kwargs = {}

    if figsize is None:
        figsize = (10, len(cmp_df) - 1)

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, None, 1, 1
    )

    yticks_pos, step = np.linspace(0, -1, ((cmp_df.shape[0]) * 2) - 2, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] - step * 1.5

    labels = cmp_df.index.values[1:]
    yticks_labels = [""] * len(yticks_pos)
    yticks_labels[0] = labels[0]
    yticks_labels[2::2] = labels[1:]

    _, ax = plt.subplots(1, figsize=figsize)

    ax.errorbar(
        x=cmp_df["elpd_loo"][1:],
        y=yticks_pos[::2],
        xerr=cmp_df.se[1:],
        label="ELPD submodels",
        color=plot_kwargs.get("color_eldp", "k"),
        fmt=plot_kwargs.get("marker_eldp", "o"),
        mfc=plot_kwargs.get("marker_fc_elpd", "white"),
        mew=linewidth,
        lw=linewidth,
    )
    ax.errorbar(
        x=cmp_df["elpd_loo"].iloc[1:],
        y=yticks_pos[1::2],
        xerr=cmp_df.dse[1:],
        label="ELPD difference",
        color=plot_kwargs.get("color_dse", "grey"),
        fmt=plot_kwargs.get("marker_dse", "^"),
        mew=linewidth,
        elinewidth=linewidth,
    )

    ax.axvline(
        cmp_df["elpd_loo"].iloc[0],
        ls=plot_kwargs.get("ls_reference", "--"),
        color=plot_kwargs.get("color_ls_reference", "grey"),
        lw=linewidth,
        label="ELPD reference",
    )

    if legend:
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            ncol=1,
            fontsize=ax_labelsize * 0.75,
        )

    if title:
        ax.set_title(
            "Model comparison\n(higher is better)",
            fontsize=ax_labelsize,
        )

    ax.set_yticks(yticks_pos)
    ax.set_xlabel("ELPD (log)", fontsize=ax_labelsize)
    ax.set_ylabel("Submodel size", fontsize=ax_labelsize)
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-1 + step, 0 - step)
    ax.tick_params(labelsize=xt_labelsize)

    return ax
