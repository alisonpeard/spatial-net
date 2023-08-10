import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import colorsys
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatial_nets import utils
from spatial_nets import PValues, LocationsDataClass


# default_cm = gt.default_cm  # 'Set3'
# The colors below come from plt.get_cmap('Set3').colors

set3_colors = list(plt.get_cmap("Set3").colors)
set3_colors.pop(1)  # the same as Peixoto does in graph-tool

default_cm = colors.LinearSegmentedColormap.from_list("graphtool-Set3", set3_colors)

default_names = [
    "aqua",
    "melrose",
    "salmon",
    "shakespeare",
    "rajah",
    "sulu",
    "classic_rose",
    "gainsboro",
    "plum",
    "snowy_mint",
    "witch_haze",
]

named_colors = dict(zip(default_names, set3_colors))


def display_cmap(cmap, N=20, ax=None):
    """Display a colormap using N bins."""
    x = np.linspace(0, 1, N)
    x = np.vstack((x, x))

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(x, cmap=cmap)
    ax.axis("off")
    ax.set_title(cmap.name)

    return None


def hsv_cmap_from_color(color, min_value, max_value, name):
    """Create a colormap from a given seed color in HSV color space."""
    h, s, v = colors.rgb_to_hsv(color)
    dark = colors.hsv_to_rgb((h, s, min_value))
    light = colors.hsv_to_rgb((h, s, max_value))
    cmap = colors.LinearSegmentedColormap.from_list(name + "_hsv", [dark, light])

    return cmap


def hls_cmap_from_color(color, min_value, max_value, name):
    """Create a colormap from a given seed color in HLS color space."""
    h, l, s = colorsys.rgb_to_hls(*color)
    dark = colorsys.hls_to_rgb(h, min_value, s)
    light = colorsys.hls_to_rgb(h, max_value, s)
    cmap = colors.LinearSegmentedColormap.from_list(name + "_hls", [dark, light])

    return cmap


def setup_default_colormaps(register=True):
    """Create the HSV and HLS colormaps based on the default Set3 colors."""
    out = {}

    for k, c in enumerate(set3_colors):
        hsv = hsv_cmap_from_color(c, 0.5, 1.0, default_names[k])
        out[hsv.name] = hsv

        hls = hls_cmap_from_color(c, 0.5, 1.0, default_names[k])
        out[hls.name] = hls

        if register:
            cm.register_cmap(name=hsv.name, cmap=hsv)
            cm.register_cmap(name=hls.name, cmap=hls)

            # also register reversed colormaps
            cm.register_cmap(name=hsv.name + "_r", cmap=hsv.reversed())
            cm.register_cmap(name=hls.name + "_r", cmap=hls.reversed())

    return out


def gt_color_legend(
    state, comms=None, ax=None, norm=None, cmap=default_cm, legendsize=(6, 0.35)
):
    """Axis with discrete colors corresponding to BlockState object."""
    if comms is not None:
        comms = np.unique(comms)
        B = len(comms)
    else:
        B = state.get_nonempty_B()  # previously I used: state.get_B()

    data = np.arange(B).reshape(1, B)

    if ax is None:
        _, ax = plt.subplots(figsize=legendsize, squeeze=True)

    ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks(range(B))

    if comms is not None:
        ax.get_xaxis().set_ticklabels(comms)

    return None


def signed_scatterplot(
    locs: LocationsDataClass,
    T_model: np.ndarray,
    pvals: PValues,
    ax,
    alpha: float = 0.2,
    fs: float = 18,
    verbose: bool = False,
    #  rounded=False,
    #  threshold=None,
):
    data = locs.flow_data
    pvals.verbose = verbose  # useful for debugging
    sig_plus, sig_minus = pvals.compute_backbone()
    zero = pvals.compute_not_significant()

    plus_observed = np.asarray(data[sig_plus.nonzero()]).flatten()
    plus_predicted = T_model[sig_plus.nonzero()]

    minus_observed = np.asarray(data[sig_minus.nonzero()]).flatten()
    minus_predicted = T_model[sig_minus.nonzero()]

    zero_observed = np.asarray(data[zero.nonzero()]).flatten()
    zero_predicted = T_model[zero.nonzero()]

    #  if rounded:
    #      plus_predicted = plus_predicted.round()
    #      minus_predicted = minus_predicted.round()
    #      zero_predicted = zero_predicted.round()

    #  if threshold:
    #      plus_predicted[plus_predicted < threshold] = 0.0
    #      minus_predicted[minus_predicted < threshold] = 0.0
    #      zero_predicted[zero_predicted < threshold] = 0.0

    ax.plot(minus_observed, minus_predicted, ".", alpha=alpha, label="left")
    ax.plot(plus_observed, plus_predicted, ".", alpha=alpha, label="right")
    ax.plot(zero_observed, zero_predicted, ".", alpha=alpha, label="not significant")

    x, X, y, Y = ax.axis()
    mM = max(x, y), min(X, Y)
    ax.plot(mM, mM, ls="--", c=".3", alpha=0.5, label="y=x (diagonal)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Observed flow", fontsize=fs)
    ax.set_ylabel("Predicted flow", fontsize=fs)

    return None


def critical_enveloppes(
    locs: LocationsDataClass,
    T_model: np.ndarray,
    pvals: PValues,
    ax,
    alpha: float = 0.5,
    verbose: bool = False,
):
    data = locs.flow_data
    pvals.verbose = verbose  # useful for debugging
    sig_plus, sig_minus = pvals.compute_backbone()

    p_obs = np.asarray(data[sig_plus.nonzero()]).flatten()
    p_pred = T_model[sig_plus.nonzero()]
    p_df = pd.DataFrame({"x": p_obs, "y": p_pred})
    p_df = p_df.groupby("x")["y"].max().sort_index()
    idx = ~(p_df < p_df.cummax())
    top = p_df.loc[idx]

    m_obs = np.asarray(data[sig_minus.nonzero()]).flatten()
    m_pred = T_model[sig_minus.nonzero()]
    m_df = pd.DataFrame({"x": m_obs, "y": m_pred})
    m_df = m_df.groupby("x")["y"].min().sort_index()[::-1]
    idx = ~(m_df > m_df.cummin())
    bottom = m_df.loc[idx]

    ax.plot(top, ls="-", color="0.3", alpha=alpha, label=r"$H_0^*$ boundary")
    ax.plot(bottom, ls="-", color="0.3", alpha=alpha)

    return top, bottom


def signed_distance_histogram(
    locs: LocationsDataClass,
    pvals: PValues,
    ax,
    nbins=20,
    alpha=0.4,
    fs=18,
    verbose=False,
):

    dmat = locs.dmat
    ds = dmat[np.triu_indices_from(dmat, k=1)]
    m, M = ds.min(), ds.max()
    bins = np.logspace(np.log10(m), np.log10(M), nbins + 1)

    pvals.verbose = verbose  # useful for debugging
    sig_plus, sig_minus = pvals.compute_backbone()
    zero = pvals.compute_not_significant()

    ds_plus = dmat[sig_plus.nonzero()]
    ds_minus = dmat[sig_minus.nonzero()]
    ds_zero = dmat[zero.nonzero()]

    ax.hist(ds_minus, bins, alpha=alpha, label="left")
    ax.hist(ds_plus, bins, alpha=alpha, label="right")
    ax.hist(ds_zero, bins, alpha=alpha, label="not significant")
    ax.set_xscale("log")

    #  ax.set_xlabel("Distance", fontsize=fs)
    #  ax.set_ylabel("Edge counts", fontsize=fs)

    return None


def contourf(x, y, z, ax, fig, labels=None, colorbar=True, **kwargs):
    im = ax.contourf(x, y, z, **kwargs)

    labels = utils._get_iterable(labels)
    if len(labels) > 1:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)

    if colorbar:
        fig.colorbar(im, cax=cax)
        if len(labels) == 3:
            cax.set_ylabel(labels[2], labelpad=10)
    else:
        cax.set_visible(False)

    return fig, ax


def selected_comms(
    state,
    comms,
    coords,
    ax=None,
    background=False,
    cmap=default_cm,
    ms=1,
    hms=10,
    legend=False,
):
    comms = np.unique(comms)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
        out = fig, ax
    else:
        out = None

    if background:
        ax.scatter(coords[:, 0], coords[:, 1], s=ms, c="k")

    if len(comms) > 1:
        bounds = np.linspace(0, len(comms), len(comms) + 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
    else:
        norm = None

    for i, c in enumerate(comms):
        idx = state.b.a == c
        color_vec = np.ones(idx.sum(), dtype=int) * i
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=hms,
            c=color_vec,
            cmap=cmap,
            norm=norm,
        )

    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        gt_color_legend(state, comms=comms, ax=cax, norm=norm)

    return out


def comm_sizes(*states, labels=None, ax=None, width=0.8, alpha=0.4):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        out = fig, ax
    else:
        out = None

    for state in states:
        u, cs = np.unique(state.b.a, return_counts=True)
        idx = np.argsort(cs)[::-1]
        ax.bar(u, cs[idx], align="center", width=width, alpha=alpha)

    ax.set_label("Community")
    ax.set_ylabel("Nb of nodes")

    if labels:
        ax.legend(labels)

    return out
