"""Autocorrelation analysis of neighborhoods."""

import datetime
import functools
import logging
import lzma
import math
from typing import Callable, List, Optional, Tuple

import colorcet as cc
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import scipy.optimize
import scipy.signal
import scipy.stats
import statsmodels.api as sm

import config
from graph_transforms.timeslice_graph import shade_attack
import util

log = logging.getLogger(__name__)

def fixed_width_vectors(times: np.array, labels: np.array, width: float = 900,
                        offset: float = 0) -> Tuple[np.ndarray, np.array]:
    """Produce vectors from bins of `width` in `labels`. All bins are offset
    such that the first bin begins at `times.min() + offset` (thus, `offset`
    must be in (`-width`, 0], where an `offset` of 0 means the first bin
    begins exactly at the earliest data point). These vectors can be used to
    compare behaviors: the dot product of normalized vectors gives one measure
    of their similarity (in terms of behaviors seen), while their magnitudes
    are another measure (in terms of the amount of behaviors seen within a
    time period).

    Return both the vector per bin as a (B x L) matrix as well as the
    number of outliers per bin as a (B x 1) array, where L = # labels, B = #
    bins."""
    t0 = times.min()
    assert -width < offset <= 0
    n_edge_bins = 1 if offset == 0 else 2
    n_bins = math.floor((times.max() - t0) / width) + n_edge_bins
    n_labels = int(labels.max() + 1) # first label is 0
    vecs = np.zeros((n_bins, n_labels)) # n_bins columns x n_labels rows
    outliers = np.zeros(n_bins)

    for t, y in tqdm.tqdm(zip(times, labels), total=len(times)):
        b = math.floor((t - t0 - offset) / width)
        y = int(y) # default NumPy type seems to be double, convert now

        if y >= 0:
            vecs[b, y] += 1
        elif y == -1:
            outliers[b] += 1

    return vecs, outliers

def make_n_label_signal(times: np.array, labels: np.array,
                        width: float = 900) -> np.array:
    """Generate bins of `width` (default: 15 minutes) containing the number of
    distinct behaviors (NOT total number of events) per bin."""
    log.info("generating small bins to create regular time series...")
    # regularization is not strictly necessary, but it's also unclear how the
    # raw data can be used in vector form to discover periodicity of behaviors
    vecs, outliers = fixed_width_vectors(times, labels, width)
    y = np.array([functools.reduce(lambda acc, x: acc + max(min(1, x), 0), vec, 0)
                  for vec in vecs])
    # apply smoothing via Savitzky-Golay (not sure what parameters to use)
    #y = scipy.signal.savgol_filter(y, 64, 4)
    return y

def make_binned_t(times: np.array, y: np.array, width: float) -> np.array:
    t0 = times.min()
    return np.arange(t0, t0 + width * len(y), width)

def autocorrelate_label(t: np.array, y: np.ndarray) \
        -> Tuple[float, float, float]:
    """Autocorrelate `y`, which is intended to be a particular row in `vecs`
    corresponding to one label. Return the acf, z-scores, and p-values."""
    acf, conf, *_ = sm.tsa.stattools.acf(y, nlags=len(t), alpha=0.05)

    # autocorrelation is nan for constant functions (i.e., horizontal lines)
    # While this is mathematically expected, it doesn't quite fit our specific
    # goal. We actually want this to have an autocorrelation of 1 (you can
    # perfectly predict the next lag given the current lag, since it's the
    # same).
    acf[np.isnan(acf)] = 1

    # lifted from statsmodels
    nobs = t.shape[0]
    varacf = np.ones_like(acf) / nobs
    varacf[0] = 0
    varacf[1] = 1.0 / nobs
    varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)

    z = acf / np.sqrt(varacf)
    p = scipy.stats.norm.sf(abs(z)) * 2

    return acf, z, p

    #max_i = np.argmin(p[start_i:]) + start_i
    #max_acf = acf[max_i]
    #max_z = z[max_i]
    #min_p = p[max_i]
    #period = t[max_i]

# this technically breaks for leap hours/seconds but oh well
def hour_formatter(x: float, pos) -> str:
    return f"{x / 3600:.0f}"

def avg_autocorrelation(times: np.array, labels: np.array,
                        plot_extra: Optional[Callable[["plt"], None]] = None,
                        width: float = 900, offset: float = 0,
                        plot_name: Optional[str] = None) -> float:
    """Compute the average autocorrelation of the given labels. Return the
    best period for this dataset."""
    log.info("creating vectors of width %d s and offset %d...", width, offset)
    vecs, outliers = fixed_width_vectors(times, labels, width, offset)
    t = make_binned_t(times, vecs, width)

    t -= t.min()

    #unit_vecs = vecs.copy()
    #for vec in unit_vecs:
    #    norm = np.linalg.norm(vec)
    #    if norm != 0:
    #        vec /= norm

    _, n_labels = vecs.shape
    acfs = []
    zs = []
    ps = []
    for label in range(n_labels):
        acf, z, p = autocorrelate_label(t, vecs[:, label])
        acfs.append(acf)
        zs.append(z)
        ps.append(p)
    acfs = np.array(acfs)
    zs = np.array(zs)
    ps = np.array(ps)

    # pick the tallest peak after the first crossover to 0
    # (Basically, try to pick the tallest peak that isn't just the unshifted
    # signal)
    avg_acfs = acfs.mean(axis=0)
    crossover = (avg_acfs <= 0).argmax()
    i_period = avg_acfs[crossover:].argmax() + crossover
    period = t[i_period]
    r_period = avg_acfs[i_period]

    if plot_name is not None:
        fig, ax = plt.subplots()
        ax.plot(t, avg_acfs)

        if t.max() < 86400:
            step = 3600 # tick every 1 h
        else:
            step = 86400 # tick every 24 h
        ax.xaxis.set_major_formatter(hour_formatter)
        ax.set_xticks(np.arange(0, t.max() + 1, step))
        ax.set_xlim(0, t.max())
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_ylim(-1, 1)
        ax.margins(x=0)
        ax.set_xlabel("Lag (h)")
        ax.set_ylabel("Average ρ")

        # annotate the max coefficient at period
        ax.plot([period], [r_period],
                color="tab:blue", marker="o", markerfacecolor="none")
        ax.annotate(f"({period/3600:.2f}, {r_period:.3f})", xy=(period, r_period),
                    xytext=(4, 4), textcoords="offset points")

        if plot_name.startswith("all"):
            i_max = 5
            period_best = t[i_max]
            r_period_best = avg_acfs[i_max]
            ax.plot([period_best], [r_period_best],
                    color="tab:red", marker="o", markerfacecolor="none")
            ax.annotate(f"({period_best/3600:.2f}, {r_period_best:.3f})", xy=(period_best, r_period_best),
                        xytext=(4, 4), textcoords="offset points")

        if plot_extra is not None:
            plot_extra(ax)

        plt.savefig(f"{plot_name}_r.pdf")
        plt.close(fig)

    return period, r_period, i_period, acfs

def nbr_autocorrelation(times: np.array, labels: np.array,
                        plot_extra: Optional[Callable[["plt"], None]] = None,
                        width: float = 900, offset: float = 0,
                        plot_name: Optional[str] = None) -> float:
    """Mimic the neighborhood plot, but plot autocorrelation coefficient and
    period on the x-axis instead."""
    log.info("creating vectors of width %d s and offset %d...", width, offset)
    vecs_, outliers = fixed_width_vectors(times, labels, width, offset)
    t_ = make_binned_t(times, vecs_, width)
    t_ -= t_.min()

    # trim off startup behavior: remove first bin
    vecs = vecs_[1:, :]
    t = t_[1:]
    n_bins, n_labels = vecs.shape

    # trim off zeros from start
    # autocorrelation should start from the first instance of a neighborhood
    shift = []
    for l in range(n_labels):
        y = vecs[:, l]
        nz = np.flatnonzero(y > 0)
        if nz.size == 0:
            # no train neighborhoods
            shift.append(None)
        else:
            i = nz.min()
            # require at least N bins, not exactly sure what N is appropriate
            # arbitrarily choose 30 bins = 7 hours for 900 second bins
            if n_bins - i < 30:
                shift.append(None)
            else:
                shift.append(i)

    acfs = []
    r_periods = []
    periods = []
    for l in range(n_labels):
        if shift[l] is None:
            acfs.append(np.array([1]))
            periods.append(-1)
            r_periods.append(0)
            continue
        # abuse of API here: as long as len(t_train) as correct, the actual
        # values don't matter. The "correct" input would be
        #   t_train[0:len(t_train) - shift[l]]
        y = vecs[shift[l]:, l]
        t_y = t[shift[l]:]
        if len(y) <= 1:
            acfs.append(np.array([1]))
            periods.append(-1)
            r_periods.append(0)
            continue
        acf, _, _ = autocorrelate_label(t_y, y)

        # use first crossover acf
        crossover = (acf <= 0).argmax() # gets first instance of True
        i = acf[crossover:].argmax() + crossover

        acfs.append(acf)
        periods.append(t_y[i])
        r_periods.append(acf[i])
    acfs = np.array(acfs)
    periods = np.array(periods)
    r_periods = np.array(r_periods)

    # scale periods to hours instead of seconds
    periods /= 3600.

    # dummy y value for labels
    y = np.arange(0, n_labels)

    # use same coloring as neighborhoods. This ends up being ordered by first
    # appearance and not neighborhood, and would probably be nice to change,
    # but oh well
    neighborhoodset = {l: None for l in labels}
    n_neighborhoods = len(neighborhoodset)
    colors = {l: cc.glasbey_dark[i % len(cc.glasbey_dark)]
              for i, l in zip(range(n_neighborhoods),
                              neighborhoodset)}

    if plot_name is not None:
        fig, ax = plt.subplots()

        for i, r in enumerate(r_periods):
            c = colors[i]
            ax.plot(r, y[i], linestyle="", marker="o", markersize=2,
                    markerfacecolor=c, markeredgecolor=c)

        #ax.plot(r_periods, y, linestyle="", marker="o", markersize=2)

        #ax2 = ax.twiny()
        #ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
        #ax2.plot(periods, y, linestyle="", marker="o", markersize=2)

        ax.set_ylim(y.min(), y.max())
        ax.set_ylabel("Provenance Neighborhood")

        ax.set_xlim(-1, 1)
        ax.set_xlabel("ρ")
        #ax2.set_xlim(0, 24 * 2) # hardcoded limit of 2 days
        #ax2.set_xlabel("Period (h)")

        if plot_extra is not None:
            plot_extra(ax)

        plt.savefig(f"{plot_name}_nbr_periods.pdf")
        plt.close(fig)

    # weighted average autocorrelation coefficient
    numerator = 0
    denominator = 0
    for l, r in enumerate(r_periods):
        n_nbrs = vecs[:, l].sum()
        numerator += n_nbrs * r
        denominator += n_nbrs

    r = numerator / denominator
    log.info("Weighted average ρ = %.3f", r)

def plot_2d_periodicity(cfg, set_name: str, fullname: Optional[str] = None,
                        n_hops: int = 1) -> None:
    if cfg == config.streamspot:
        width = 180 # shorter, so use smaller (3 min) bins
    else:
        width = 900

    labels_path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop.pkl.xz"
    log.info("reading %s...", labels_path)
    labels, times, *_ = util.pickle_read(labels_path)

    if fullname is None:
        fullname = set_name

    shade_attack_periods = shade_attack(cfg, set_name)
    out = avg_autocorrelation(times, labels,
                              width=width,
                              plot_extra=shade_attack_periods,
                              plot_name=f"{fullname}_2d_periodicity")

    return out

def period_plotter(period: float, times: np.array) -> Callable[["plt"], None]:
    def plotter(plt: "matplotlib.pyplot") -> None:
        tmax = times.max()
        t = times.min()
        while t < tmax + period:
            plt.axvline(t, color="limegreen", linestyle="--")
            t += period

    return plotter

def plot_set_periods(cfg, set_name: str, fullname: Optional[str] = None,
                     n_hops: int = 1) -> None:
    labels_path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop.pkl.xz"

    log.info("reading %s...", labels_path)
    labels, times, *_ = util.pickle_read(labels_path)

    if cfg == config.streamspot:
        #width = 180 # shorter, so use smaller (3 min) bins
        width = 100 # fix bin width to exact trial length
    else:
        width = 900

    if fullname is None:
        fullname = set_name

    shade_attack_periods = shade_attack(cfg, set_name)

    log.info("Plotting 2D autocorrelation...")
    plot_name = f"{fullname}_{n_hops}hop"
    period, rho, *_ = avg_autocorrelation(times, labels, width=width,
                                          plot_extra=shade_attack_periods,
                                          plot_name=plot_name)
    log.info("avg: period = %s, ρ = %s", period, rho)
    nbr_autocorrelation(times, labels, width=width,
                        plot_extra=None, plot_name=plot_name)

def main(argv: List[str]) -> None:
    for n_hops in [1, 2]:
        for set_name in ["cadets", "theia", "fivedirections", "trace", "trace_departitioned"]:
            plot_set_periods(config.e3, set_name, n_hops=n_hops)

        for set_name in ["h1", "h2"]:
            set_name = f"{set_name}_benign"
            plot_set_periods(config.atlasv2, set_name, n_hops=n_hops)

        for set_name in ["host201", "host501"]:
            plot_set_periods(config.optc, set_name, n_hops=n_hops)

        for set_name in ["all"]:
            plot_set_periods(config.streamspot, set_name, n_hops=n_hops)
