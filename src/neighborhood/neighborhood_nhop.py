import csv
import datetime
import functools
import itertools
import logging
import lzma
import math
import tempfile
import sys
import uuid
from typing import Tuple, Set, List, Optional, Callable, TextIO

import brokenaxes
import colorcet as cc
import igraph as ig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics
import tqdm

import config
import gnn.util.benign_day_edges
import graph_transforms.reverse_dns
import graph_transforms.ip_whois
import graph_transforms.sanitize_proc_names
import parse.graph
import util
from graph_transforms.timeslice_graph import shade_attack
from name_templates import Templatizer

log = logging.getLogger(__name__)

def node_name(templatizer: Templatizer, u: ig.Vertex,
              template_tracker: Optional[dict] = None) -> str:
    """Get the name of a node `u` according to its name field and any possible
    corresponding templates in `templatizer`.

    The optional `template_tracker` is a `str -> set[str]` mapping templates
    to all unique names they contain."""
    name = u["name"]
    if name.startswith(parse.graph.unknown_prefix): # use same embedding
        return name[:5]
    # XXX temporarily turn off sockets
    if u["type"] == "socket":
        return "!!!socket!!!"
    if u["type"] == "process":
        out = graph_transforms.sanitize_proc_names.sanitize_name(
                name, templatizer.path_separator)
    else:
        t = templatizer.get_template(name)
        if t is not None:
            if template_tracker is not None:
                if t not in template_tracker:
                    template_tracker[t] = set()
                template_tracker[t].add(name)
            out = t
        else:
            out = name
    # ignore case
    return out.lower()

class CtrMapping:
    """Convenience class that maps objects to ints using a counter."""
    def __init__(self):
        self.m = {}
        self.ctr = 0

    def map(self, obj) -> Tuple[int, bool]:
        """Map `obj`, and return if this is the first time `obj` was seen."""
        new = obj not in self.m
        if new:
            self.m[obj] = self.ctr
            self.ctr += 1
        return self.m[obj], new

    def __getitem__(self, obj) -> int:
        """Same as `map`, but don't return whether the object is new."""
        return self.map(obj)[0]

def zerohop_neighborhoods(g: ig.Graph, templatizer: Templatizer,
                          mapping: CtrMapping = CtrMapping(),
                          template_tracker: Optional[dict] = None) -> np.array:
    """Produce the "zero-hop" neighborhoods of `g`. This is a slightly special
    case to allow for different definitions of node equality. Currently, nodes
    are equal if they have the same name.

    `mapping` can be reused to ensure different graphs share the same neighbor
    label assignments.

    Note that the overall algorithm is the same as nhop --- it would be
    possible to make nhop extend to 0hop, but would make parameterization
    unnecessarily awkward."""
    # TODO typing information? right now procs and files are equivalent
    n = g.vcount()
    nbrs = np.full(n, -1, dtype=np.int)
    for u in tqdm.tqdm(g.vs, total=n):
        name = node_name(templatizer, u, template_tracker)
        nbrs[u.index] = mapping[name]
    assert -1 not in nbrs
    return nbrs

def npissubset(ar1: np.array, ar2: np.array) -> bool:
    """Return whether `ar1` is a subset of `ar2`. This assumes both `ar1`
    and `ar2` are unique."""
    return len(np.setdiff1d(ar1, ar2, assume_unique=True)) == 0

def replace_subsets(nbrs: np.array, sets: dict) -> None:
    """Usings `sets`, find the largest superset per neighbor in `nbrs` and
    replace the neighbor id with the superset id."""
    for root, subsets in sets.items():
        # ascending sort, then iterate in reverse
        subsets = sorted(subsets, key=lambda t: (len(t[0]), *t[0]))
        while subsets:
            sup, supembed = subsets.pop()
            buf = []
            for sub, subembed in subsets:
                if npissubset(sub, sup):
                    # replace all sub embeddings with sup embedding
                    nbrs[nbrs == subembed] = supembed
                else:
                    buf.append((sub, subembed))
            subsets = buf

def nhop_neighborhoods(adj: List[List[int]], prev: np.array,
                       mapping: CtrMapping = CtrMapping(),
                       sets: dict = {},
                       do_subsets: bool = True) -> np.array:
    """Given the adjacency list `adj` and the N-1 hop neighborhoods `prev`,
    output the N hop neighborhoods. This algorithm places subsets of a
    particular neighborhood into the same label as the largest superset.

    Assignments from another graph can be shared by passing `mapping` and
    `sets`."""
    # this propagates file neighborhoods; another strategy is to keep those
    # constant at the "0-hop embedding"
    nbrs = np.full_like(prev, -1)
    # generate sets and initial mapping first
    for u, vs in enumerate(tqdm.tqdm(adj)):
        if prev[u] not in sets:
            sets[prev[u]] = []
        s = np.unique(prev[vs])
        embedding = (prev[u], *s)
        nbrid, new = mapping.map(embedding)
        if new:
            sets[prev[u]].append((s, nbrid))
        nbrs[u] = nbrid

    assert -1 not in nbrs
    if do_subsets:
        replace_subsets(nbrs, sets)

    return nbrs

def make_proc_mapping(g: ig.Graph) -> np.array:
    """Make a mapping of each node in `g` to whether it should be considered a
    process. This mostly exists to handle execution units or threads, which
    currently have the "unknown" type."""
    n = g.vcount()
    arr = np.full(n, False, dtype=bool)
    for u in tqdm.tqdm(g.vs, total=n):
        if u["type"] == "process":
            arr[u.index] = True
        elif u["type"] == "unknown":
            # look for process behavior
            out_proc_events = {"clone", "write", "send", "sendto", "sendmsg"}
            in_proc_events = {"clone", "read", "recv", "recvfrom", "recvmsg"}
            arr[u.index] = (
                len(g.es(_source=u.index, type_in=out_proc_events)) > 0
                or len(g.es(_target=u.index, type_in=in_proc_events)) > 0
            )
    return arr

def adjust_nbr_sockets(g: ig.Graph, adj):
    for u, vs in enumerate(tqdm.tqdm(adj)):
        min_t = np.inf
        # only consider network sockets, not UNIX domain sockets
        def is_net(u: ig.Vertex):
            # prune off `sock=`
            name = u["name"][5:]
            return not (name.startswith('"') or name.startswith("/")
                        or name.startswith("\\")
                        or name == "localhost")

        sockets = g.vs[vs].select(type_eq="socket").select(is_net)
        #times = {}
        #for skt in tqdm.tqdm(sockets, leave=False):
        #    t = min(g.es(_within=[skt.index, u])["time"])
        #    times[skt.index] = t

        # this might be slightly less accurate, but should be much faster
        times = {skt.index: skt["time"] for skt in sockets}

        # only keep the first N sockets
        keep_n = 2
        rm = sorted(times.items(), key=lambda t: t[1])[keep_n:]
        for skt, _ in rm:
            vs.remove(skt)

def proc_neighborhoods(g: ig.Graph, templatizer: Templatizer, n_hops: int,
                       adjust_sockets: bool = False,
                       time_scale: float = 1e9,
                       time_cutoff = datetime.datetime(2018, 4, 1)):
    """Strategy: an N-hop neighborhood can be represented as a 1-hop
    neighborhood of (N - 1)-hop neighborhoods. We can compute this in
    N * |V| * degree(V), which should still be reasonable at low N.

    Note that igraph's default representation is actually just a node and edge
    list, which is why neighbor queries are so slow for larger graphs. It's
    much better to first simplify the graph (removing parallel edges), then
    get the graph in adjacency list format.

    Return three arrays corresponding to process nodes in g:
        times, the time of each process node
        g_node_ids, the node ID of each process node
        labels, the neighborhood ID of each process node
        nbrs, the list of i-hop neighborhood embeddings for all i in n_hops

    Note that this doesn't actually output the neighborhoods in graph form, as
    they are essentially being represented as hashes instead.

    NOTE: due to memory usage concerns, this modifies `g` in-place instead of
    copying it."""

    log.info("deciding which nodes are processes...")
    proc_mapping = make_proc_mapping(g)
    g_node_ids = np.flatnonzero(proc_mapping)

    # simplify the graph now: don't need parallel edges
    log.info("removing parallel edges...")
    g.simplify(multiple=True, loops=False, combine_edges="first")

    log.info("computing adjacency list...")
    adj = g.get_adjlist(mode="in")

    if adjust_sockets:
        log.info("adjusting adjacency list sockets...")
        adjust_nbr_sockets(g, adj)

    if template_tracker is not None:
        log.info("tracking templates!")

    log.info("computing 0-hop neighborhoods")
    nbrs = [zerohop_neighborhoods(g, templatizer, template_tracker=template_tracker)]

    # iterate up to n
    for i in range(1, n_hops + 1):
        log.info("computing %s-hop neighborhoods", i)
        nbrs.append(nhop_neighborhoods(adj, nbrs[-1]))

    # prune out the non-process neighborhoods
    labels = nbrs[-1][g_node_ids]

    # compute times
    log.info("creating times array...")
    times = np.zeros_like(g_node_ids, dtype=float)
    for i, nid in enumerate(g_node_ids):
        times[i] = g.vs[nid]["time"] / time_scale

    util.adjust_zero_times(times, time_cutoff.timestamp())

    labels = adjust_neighbors(times, labels)

    if template_tracker is not None:
        out_path = "templates.pkl"
        log.info("writing tracked templates to %s", out_path)
        util.pickle_write(template_tracker, out_path)

    return (times, g_node_ids, labels, nbrs)

def adjust_neighbors(times: np.array, nbrs: np.array) -> np.array:
    """Adjust `nbrs` so that new labels increase monotonically with time
    without any gaps."""
    # currently keeping singletons, unlike previous versions
    unique = np.unique(nbrs)
    log.info("total of %s unique labels", len(unique))

    log.info("renumbering labels based on timestamp...")
    earliest_times = {}
    for label in unique:
        earliest_times[label] = times[nbrs == label].min()

    labels = nbrs.copy()
    for new_label, (label, _) in enumerate(sorted(earliest_times.items(),
                                                  key=lambda t: t[1])):
        labels[nbrs == label] = new_label

    return labels

def to_hour(timestamp: float, tz: datetime.timezone) -> datetime.datetime:
    """Return a datetime of `timestamp` truncated to the nearest hour."""
    dt = datetime.datetime.fromtimestamp(timestamp, tz=tz)
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, tzinfo=tz)

def make_hour_formatter(tz: datetime.timezone) -> Callable:
    return lambda x, pos: datetime.datetime.fromtimestamp(x, tz=tz).strftime("%m-%d %H")

def plot_neighbors(cfg, labels: np.array, times: np.array,
                   plot_extra: Optional[Callable[["plt"], None]] = None,
                   out_name: str = "labels",
                   tz: datetime.timezone = util.UTC,
                   needs_brokenaxes: bool = False) -> None:
    """Plot the neighborhood labels of `G`: output labels vs time to
    `out_name`.pdf."""
    # how this works:
    # nodes in G are ordered (G.vs[0], ..., G.vs[N] is node 0, ..., N)
    # nbrs[i] is the neighborhood label for neighborhood i

    # use dict to maintain order (requires Python >= 3.7)
    labelset = {l: None for l in labels}
    neighborhoodset = labelset.copy()

    n_processes = len(labels)
    n_neighborhoods = len(neighborhoodset)

    np.set_printoptions(threshold=sys.maxsize)
    log.info("number of processes: %s", f"{n_processes:,}")
    log.info("number of distinct neighborhoods: %s", f"{n_neighborhoods:,}")

    colors = {l: cc.glasbey_dark[i % len(cc.glasbey_dark)]
              for i, l in zip(range(n_neighborhoods),
                              neighborhoodset)}

    plt.clf()

    # generate broken axis, if necessary
    # NOTE: plotting on brokenaxes is extremely slow, so not recommended
    # unless strictly necessary
    if needs_brokenaxes:
        sorted_times = np.sort(times)
        # also force each bin to span from day to day
        max_gap = 2 * 3600 # maximum gap of 2 hours between times
        lims = []
        start = to_hour(sorted_times[0], tz).timestamp()
        for t1, t2 in zip(sorted_times, sorted_times[1:]):
            next_start = to_hour(t2, tz).timestamp()
            if next_start <= start:
                continue
            end = (to_hour(t1, tz) + datetime.timedelta(hours=2)).timestamp()
            if next_start - end > max_gap:
                lims.append((start, end))
                start = next_start

        # add last lim
        lims.append((start, end))

        bax = brokenaxes.brokenaxes(xlims=lims, wspace=0.1, bottom=0.15,
                                    despine=False)
    else:
        bax = plt

    for label, color in colors.items():
        class_member_mask = labels == label
        x = times[class_member_mask]
        y = np.repeat(label, x.size)
        bax.plot(x, y, linestyle="", marker="o", markerfacecolor=color,
                 markeredgecolor=color, markersize=2,
                 rasterized=True)

    # plot any extras, e.g., trial boundaries if they exist
    if plot_extra is not None:
        plot_extra(plt)

    if needs_brokenaxes:
        # ATLASv1: decide which of the bins are attack periods
        if cfg == config.atlasv1:
            # out_name contains set_name (yes, this is a hack)
            if "h1" in out_name:
                set_name = "h1"
            elif "h2" in out_name:
                set_name = "h2"
            else:
                raise ValueError("unknown set")
            path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
            log.info("reading graph from %s...", path)
            g = util.pickle_read(path)

            # all attack periods contain "payload.exe"
            log.info("finding and shading ATLASv1 attack periods...")
            atk_lims = set()
            for atk_t in g.vs(lambda u: u["name"].endswith("\\payload.exe"))["time"]:
                atk_t /= 1e9
                for lim in lims:
                    if atk_t >= lim[0] and atk_t <= lim[1]:
                        atk_lims.add(lim)
                        break

            # shade the attack bins now
            tmin = times.min()
            tmax = times.max()
            for lim in atk_lims:
                lo = lim[0]
                hi = lim[1]
                if lo < tmin:
                    lo = tmin
                if hi > tmax:
                    hi = tmax
                shade_attack_region(lo, hi, bax)

        # need to manually set this, xaxis_format_dates won't work because
        # each axis has a copy of the data
        for (lim, ax) in zip(lims, bax.axs):
            ax.xaxis.set_major_formatter(make_hour_formatter(tz))
            start = to_hour(lim[0], tz).timestamp()
            end = to_hour(lim[1], tz).timestamp()
            ax.set_xticks(np.arange(start, end, 2 * 3600))
            ax.set_xlim(start, end)

        fig = plt.gcf()
        fig.autofmt_xdate()
        fig.set_tight_layout(False)
        bax.set_ylabel("Provenance Neighborhoods")
        bax.set_xlabel("Time")
        plt.tight_layout()
        for x in bax.diag_handles: x.remove()
        bax.draw_diags()
    else:
        plt.ylabel("Provenance Neighborhoods")
        plt.ylim(-1, max(labelset) + 1)
        if cfg == config.streamspot:
            plt.xlabel("Hours")
            # streamspot just uses a monotonic timer starting from 0
            # and is very short, so only format as hours
            ax = plt.gca()
            start = 0
            end = times.max()
            # round end to the nearest hour
            end = math.ceil(end / 3600) * 3600
            ax.xaxis.set_major_formatter(lambda x, pos: int(x / 3600))
            ax.set_xticks(np.arange(start, end, 3600))
            ax.set_xlim(start, end)
            # remove fractional y axis ticks
            #ax.set_yticks(np.arange(-1, max(labelset) + 1, 2))
        else:
            plt.xlabel("Time")
            util.xaxis_format_dates(plt.gcf(), plt.gca(), step=1, tz=tz)
            # manually force same range for host5
            #ax = plt.gca()
            #start = datetime.datetime(2019, 9, 16, tzinfo=tz).timestamp()
            #end = datetime.datetime(2019, 9, 26, tzinfo=tz).timestamp()
            #ax.set_xticks(np.arange(start, end, 86400))
            #ax.set_xlim(start, end)

    log.info("writing %s.pdf...", out_name)
    plt.savefig(f"{out_name}.pdf")

def plot_neighbors_atk(cfg, labels: np.array, times: np.array,
                       attack_mask: np.array,
                       plot_extra: Optional[Callable[["plt"], None]] = None,
                       out_name: str = "labels",
                       tz: datetime.timezone = util.UTC) -> None:
    labelset = {l: None for l in labels}

    n_processes = len(labels)
    n_neighborhoods = len(labelset)

    np.set_printoptions(threshold=sys.maxsize)
    log.info("number of processes: %s", f"{n_processes:,}")
    log.info("number of distinct neighborhoods: %s", f"{n_neighborhoods:,}")

    plt.clf()
    for label in labelset:
        class_member_mask = labels == label

        #x = times[~attack_mask & class_member_mask]
        x = times[class_member_mask]
        y = np.repeat(label, x.size)
        color = "#aaaaaa"
        plt.plot(x, y, linestyle="", marker="o", markerfacecolor=color,
                 markeredgecolor=color, markersize=2, zorder=1,
                 rasterized=True)

        x = times[attack_mask & class_member_mask]
        y = np.repeat(label, x.size)
        color = "#00ff00"
        plt.plot(x, y, linestyle="", marker="o", markerfacecolor=color,
                 markeredgecolor=color, markersize=2, zorder=2,
                 rasterized=True)

    # plot any extras, e.g., trial boundaries if they exist
    if plot_extra is not None:
        plot_extra(plt)

    plt.ylabel("Provenance Neighborhoods")
    plt.ylim(-1, max(labelset) + 1)
    plt.xlabel("Time")
    util.xaxis_format_dates(plt.gcf(), plt.gca(), step=1, tz=tz)

    out_name = f"{out_name}_atk.pdf"
    log.info("writing %s...", out_name)
    plt.savefig(out_name)

def plot_anomaly(cfg, labels: np.array, times: np.array, scores: np.array,
                 attack_mask: np.array,
                 plot_extra: Optional[Callable[["plt"], None]] = None,
                 out_name: str = "labels",
                 tz: datetime.timezone = util.UTC) -> None:
    """Plot neighborhoods colored according to their anomaly score.
    Use different shapes for attack and benign nodes.

    This requires every node to have a score; for nodes that don't inherently
    have one (e.g., training nodes), it's recommended to give them the minimum
    score, e.g., 0.
    """
    # sort by ascending score, so that anomalies are always on top
    i = scores.argsort()
    scores = scores[i]
    labels = labels[i]
    times = times[i]

    cmap = cc.m_CET_R3.copy()
    cmap.set_bad(cmap(0))

    labelset = {l: None for l in labels}

    n_processes = len(labels)
    n_neighborhoods = len(labelset)

    np.set_printoptions(threshold=sys.maxsize)
    log.info("number of processes: %s", f"{n_processes:,}")
    log.info("number of distinct neighborhoods: %s", f"{n_neighborhoods:,}")

    plt.clf()
    vmin = scores[scores > 0].min()
    vmax = scores.max()
    # don't need log scaling, anomaly space seems to be mostly linear
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    X = []
    Y = []
    C = []
    X_atk = []
    Y_atk = []
    C_atk = []
    for label in labelset:
        class_member_mask = labels == label

        mask = ~attack_mask & class_member_mask
        c = scores[mask]
        x = times[mask]
        y = np.repeat(label, x.size)
        X.append(x)
        Y.append(y)
        C.append(c)
        #plt.scatter(x, y, marker="o", s=2,
        #            c=c, cmap=cmap, norm=norm,
        #            rasterized=True)

        mask = attack_mask & class_member_mask
        c = scores[mask]
        x = times[mask]
        y = np.repeat(label, x.size)
        X_atk.append(x)
        Y_atk.append(y)
        C_atk.append(c)
        #plt.scatter(x, y, marker="D", s=2,
        #            c=c, cmap=cmap, norm=norm,
        #            rasterized=True)

    X = np.concatenate(X, axis=None)
    Y = np.concatenate(Y, axis=None)
    C = np.concatenate(C, axis=None)
    plt.scatter(X, Y, marker="o", s=2,
                c=C, cmap=cmap, norm=norm, rasterized=True)

    X_atk = np.concatenate(X_atk, axis=None)
    Y_atk = np.concatenate(Y_atk, axis=None)
    C_atk = np.concatenate(C_atk, axis=None)
    plt.scatter(X_atk, Y_atk, marker="D", s=2,
                c=C_atk, cmap=cmap, norm=norm, rasterized=True)

    # plot any extras, e.g., trial boundaries if they exist
    if plot_extra is not None:
        plot_extra(plt)

    plt.ylabel("Provenance Neighborhoods")
    plt.ylim(-1, max(labelset) + 1)
    plt.xlabel("Time")
    util.xaxis_format_dates(plt.gcf(), plt.gca(), step=1, tz=tz)

    plt.colorbar()

    out_name = f"{out_name}_anomaly.pdf"
    log.info("writing %s...", out_name)
    plt.savefig(out_name)


# maybe try stacked 1D histograms with LinearSegmentedColormap and pcolormesh?
# only problem is it might be worst of both worlds - both hard to see
# specific times, and hard to see number of events
def hist_neighbors(cfg, labels: np.array, times: np.array,
                   plot_extra: Optional[Callable[["plt"], None]] = None,
                   out_name: str = "labels_hist",
                   tz: datetime.timezone = util.UTC,
                   needs_brokenaxes: bool = False) -> None:
    """Plot the neighborhood labels of `G` as a 2D histogram. This loses the
    clear distinction of event types, but clearly shows frequency information
    that is lost by plot_neighbors. Output labels vs time to `out_name`.pdf."""
    # use dict to maintain order (requires Python >= 3.7)
    labelset = {l: None for l in labels}
    n_nbrs = labels.max() + 1

    np.set_printoptions(threshold=sys.maxsize)
    log.info("Number of neighborhoods: %s", f"{n_nbrs:,}")

    if cfg == config.atlasv1 or cfg == config.streamspot:
        bin_width = 600 # 600 s = 10 minutes
    else:
        bin_width = 3600 # 3600 s = 1 hour
    n_time_bins = int(np.ceil((times.max() - times.min()) / bin_width))
    log.info("Number of time bins: %s", n_time_bins)

    n_nbr_bins = n_nbrs
    # too few neighborhoods will also look bad, fix now
    i = 1
    if n_nbr_bins < 100:
        while n_nbr_bins < 100:
            n_nbr_bins += n_nbrs
            i += 1
        log.info("Number of neighborhood bins: %s", n_nbr_bins)
        log.info("%sX expansion", i)
    else:
        while n_nbr_bins > 500:
            # ceiling divide by 2
            n_nbr_bins = (n_nbr_bins + 1) // 2
            i *= 2
        log.info("Number of neighborhood bins: %s", n_nbr_bins)
        log.info("%sX compression", i)

    plt.clf()
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(cmap(0))

    # generate broken axis, if necessary
    # NOTE: plotting on brokenaxes is extremely slow, so not recommended
    # unless strictly necessary
    if needs_brokenaxes:
        sorted_times = np.sort(times)
        # also force each bin to span from day to day
        max_gap = 2 * 3600 # maximum gap of 2 hours between times
        lims = []
        start = to_hour(sorted_times[0], tz).timestamp()
        for t1, t2 in zip(sorted_times, sorted_times[1:]):
            next_start = to_hour(t2, tz).timestamp()
            if next_start <= start:
                continue
            end = (to_hour(t1, tz) + datetime.timedelta(hours=2)).timestamp()
            if next_start - end > max_gap:
                lims.append((start, end))
                start = next_start

        # add last lim
        lims.append((start, end))

        bax = brokenaxes.brokenaxes(xlims=lims, wspace=0.1, bottom=0.15,
                                    despine=False)
        for ax in bax.axs:
            _, _, _, qm = ax.hist2d(times, labels,
                                    cmap=cmap,
                                    bins=[n_time_bins, n_nbr_bins],
                                    #bins=100,
                                    norm=mpl.colors.LogNorm(),
                                    rasterized=True)
    else:
        _, _, _, qm = plt.hist2d(times, labels,
                                 cmap=cmap,
                                 bins=[n_time_bins, n_nbr_bins],
                                 #bins=100,
                                 norm=mpl.colors.LogNorm(),
                                 rasterized=True)

    # plot any extras, e.g., trial boundaries if they exist
    if plot_extra is not None:
        plot_extra(plt)

    if needs_brokenaxes:
        # ATLASv1: decide which of the bins are attack periods
        if cfg == config.atlasv1:
            # out_name contains set_name (yes, this is a hack)
            if "h1" in out_name:
                set_name = "h1"
            elif "h2" in out_name:
                set_name = "h2"
            else:
                raise ValueError("unknown set")
            path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
            log.info("reading graph from %s...", path)
            g = util.pickle_read(path)

            # all attack periods contain "payload.exe"
            log.info("finding and shading ATLASv1 attack periods...")
            atk_lims = set()
            for atk_t in g.vs(lambda u: u["name"].endswith("\\payload.exe"))["time"]:
                atk_t /= 1e9
                for lim in lims:
                    if atk_t >= lim[0] and atk_t <= lim[1]:
                        atk_lims.add(lim)
                        break

            # shade the attack bins now
            tmin = times.min()
            tmax = times.max()
            for lim in atk_lims:
                lo = lim[0]
                hi = lim[1]
                if lo < tmin:
                    lo = tmin
                if hi > tmax:
                    hi = tmax
                shade_attack_region(lo, hi, bax)

        fig = plt.gcf()
        fig.autofmt_xdate()

        # rasterized because vector causes a weird grid artifact
        fig.colorbar(qm, ax=bax.axs)

        # need to manually set this, xaxis_format_dates won't work because
        # each axis has a copy of the data
        for (lim, ax) in zip(lims, bax.axs):
            ax.xaxis.set_major_formatter(make_hour_formatter(tz))
            start = to_hour(lim[0], tz).timestamp()
            end = to_hour(lim[1], tz).timestamp()
            ax.set_xticks(np.arange(start, end, 2 * 3600))
            ax.set_xlim(start, end)

        for ax in bax.axs[1:]:
            ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        bax.axs[0].yaxis.set_major_locator(mpl.ticker.AutoLocator())
        bax.axs[0].yaxis.set_minor_locator(mpl.ticker.AutoLocator())

        fig.set_tight_layout(False)
        bax.set_ylabel("Provenance Neighborhoods")
        bax.set_xlabel("Time", labelpad=50)
        for x in bax.diag_handles: x.remove()
        bax.draw_diags()
    else:
        # rasterized because vector causes a weird grid artifact
        plt.colorbar(qm)

        plt.ylabel("Provenance Neighborhoods")
        if cfg == config.streamspot:
            plt.xlabel("Hours")
            # streamspot just uses a monotonic timer starting from 0
            # and is very short, so only format as hours
            ax = plt.gca()
            start = 0
            end = times.max()
            # round end to the nearest hour
            end = math.ceil(end / 3600) * 3600
            ax.xaxis.set_major_formatter(lambda x, pos: int(x / 3600))
            ax.set_xticks(np.arange(start, end, 3600))
            ax.set_xlim(start, end)
        else:
            plt.xlabel("Time")
            util.xaxis_format_dates(plt.gcf(), plt.gca(), step=1, tz=tz)
        plt.autoscale(enable=True, axis="y", tight=True)

    log.info("writing %s.pdf...", out_name)
    plt.savefig(f"{out_name}.pdf")

def write_procs_txt(g_node_ids: np.array, g: ig.Graph, out_path: str) -> None:
    """Write a sorted list of `count: name` pairs of processes in `g`."""
    name_counts = {}
    for name in g.vs(g_node_ids)["name"]:
        if name.startswith(parse.graph.unknown_prefix):
            name = parse.graph.unknown_proc
        if name not in name_counts:
            name_counts[name] = 0
        name_counts[name] += 1

    with open(out_path, "wt") as f:
        for name, count in sorted(name_counts.items(), key=lambda t: t[1],
                                  reverse=True):
            f.write(f"{count}: {name}\n")

def write_neighbors_txt(labels: np.array, nbrs: List[np.array],
                        g_node_ids: np.array, g: ig.Graph,
                        n_hops: int, out_path: str) -> None:
    """Write a sample neighborhood (of the given `n_hops`) for each label in
    text format to `out_path`."""
    unique, indices, count = np.unique(labels, return_index=True,
                                       return_counts=True)

    # precompute adjlist, since lookup is slow
    adj = g.get_adjlist(mode="in")

    def write_nbr(f: TextIO, g: ig.Graph, nbrs: List[np.array],
                  adj: List[List[int]], max_depth: int, depth: int, root: int):
        # start at (N - 1)-hop, end at 0
        name = g.vs[root]["name"]
        inv_depth = max_depth - depth
        f.write(f"{' ' * 2 * inv_depth}└─{name} [{root}]\n")

        depth -= 1
        if depth < 0:
            return

        embeddings = nbrs[depth]
        nbr = adj[root]
        _, indices = np.unique(embeddings[nbr], return_index=True)
        for i in indices:
            write_nbr(f, g, nbrs, adj, max_depth, depth, nbr[i])

    with open(out_path, "wt") as f:
        for i in tqdm.tqdm(np.argsort(count)[::-1]):
            l = unique[i]
            g_i = indices[i]
            n_i = count[i]

            assert l == labels[g_i]
            root = g_node_ids[g_i]

            f.write(f"label {l}: n = {n_i}\n")
            write_nbr(f, g, nbrs, adj, n_hops, n_hops, root)
            f.write("\n")

def make_attack_mask_nids(g_node_ids: np.array, g: ig.Graph,
                          attack_node_ids: Set[int]) -> np.array:
    """Return the mask for `g_node_ids` that only selects nodes with node IDs
    in `attack_node_ids`. (This particular labeling scheme is used for
    streamspot)."""
    log.info("constructing attack node mask...")
    mask = np.full(len(g_node_ids), False, dtype=bool)
    for i, nid in enumerate(g_node_ids):
        if nid in attack_node_ids:
            mask[i] = True
    return mask

def make_attack_mask_uuids(g_node_ids: np.array, g: ig.Graph,
                           attack_node_uuids: Set[str]) -> np.array:
    """Return the mask for `g_node_ids` that only selects nodes with UUIDs in
    `attack_node_uuids`."""
    log.info("constructing attack node mask...")
    mask = np.full(len(g_node_ids), False, dtype=bool)
    # this assumes that the order of iter and g.vs(iter) correspond, which may
    # not be the safest assumption...
    for i, uuid in enumerate(g.vs(g_node_ids)["uuid"]):
        if uuid in attack_node_uuids:
            mask[i] = True
    return mask

def pickle_neighborhoods(cfg, set_name: str, n_hops: int = 2,
                         collapsed: bool = False) -> None:
    fname = f"{set_name}_prov_graph"
    if collapsed:
        fname = f"{fname}_collapsed_sockets"
    graph_path = f"{cfg.graphs}/{fname}.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    if cfg == config.streamspot:
        g = g.g

    # group sockets based on domain names
    #graph_transforms.reverse_dns.rename_sockets(cfg, set_name, g)
    #graph_transforms.ip_whois.convert_sockets(g)

    if (set_name == "fivedirections" or cfg == config.atlasv2
        or cfg == config.atlasv1 or cfg == config.optc):
        sep = "\\"
    else:
        sep = "/"

    if cfg == config.streamspot:
        names = set(g.vs["name"])
        templatizer = Templatizer(list(names), sep)
    else:
        if set_name.endswith("_benign"):
            name_path = f"{cfg.names}/{set_name[:-len('_benign')]}_prov_graph.txt"
        else:
            name_path = f"{cfg.names}/{set_name}_prov_graph.txt"
        with open(name_path, "r") as f:
            templatizer = Templatizer(f.readlines(), sep)

    if cfg == config.e3:
        # nanosecond timestamps
        cutoff = datetime.datetime(2018, 4, 1)
        scale = 1e9
    elif cfg == config.atlasv2:
        # nanosecond timestamps
        cutoff = datetime.datetime(2022, 7, 10)
        scale = 1e9
    elif cfg == config.atlasv1:
        # nanosecond timestamps
        cutoff = datetime.datetime(2018, 1, 1)
        scale = 1e9
    elif cfg == config.streamspot:
        # no cutoff needed, first event is at t = 0
        cutoff = datetime.datetime(1969, 12, 31)
        scale = 1e9
    elif cfg == config.optc:
        # nanosecond timestamps
        cutoff = datetime.datetime(2019, 9, 1)
        scale = 1e9

    times, g_node_ids, labels, nbrs = proc_neighborhoods(
            g, templatizer, n_hops,
            adjust_sockets=cfg == config.streamspot,
            #adjust_sockets=True,
            time_scale=scale, time_cutoff=cutoff)

    fname = f"{set_name}_labels_{n_hops}hop"
    if collapsed:
        fname = f"{fname}_collapsed_sockets"
    labels_path = f"{cfg.neighborhoods}/{fname}.pkl.xz"
    log.info("saving labels to %s...", labels_path)
    util.pickle_write((labels, times, g_node_ids, nbrs), labels_path)

def shade_attack_region(start: float, end: float, plt: "matplotlib.pyplot"):
    """Shade the region from start to end as attack."""
    plt.axvspan(start, end, facecolor="#ff0000", alpha=0.7, zorder=10)

def plot_set(cfg, set_name: str, needs_brokenaxes: bool = False,
             n_hops: int = 1) -> None:
    labels_path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop.pkl.xz"

    log.info("Loading %s", labels_path)
    labels, times, g_node_ids, *_ = util.pickle_read(labels_path)

    if cfg == config.atlasv1:
        fullname = f"v1{set_name}"
    elif cfg == config.atlasv2:
        fullname = f"v2{set_name}"
    else:
        fullname = set_name

    if cfg == config.e3:
        shade_attack_periods = shade_attack(cfg, set_name)
    elif cfg == config.streamspot and set_name == "all":
        # load provenance graph
        graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
        log.info("loading ProvenanceGraph from %s...", graph_path)
        g = util.pickle_read(graph_path)

        assert len(g.trial_boundaries) == 599
        # attack: last 100 trials (but last boundary is missing)
        start = g.trial_boundaries[500] / 1e9
        end = times.max()
        shade_attack_periods = functools.partial(shade_attack_region,
                                                 start, end)
    elif cfg == config.atlasv2:
        start = 1658235694.3 # computed as last time in benign day 5
        end = times.max()
        shade_attack_periods = functools.partial(shade_attack_region,
                                                 start, end)
    else:
        shade_attack_periods = shade_attack(cfg, set_name)

    if cfg == config.e3 or cfg == config.streamspot or cfg == config.atlasv1:
        tz = util.EDT
    elif cfg == config.atlasv2:
        tz = util.CDT
    elif cfg == config.optc:
        # OpTC ground truth is all based on EDT times
        tz = util.EDT

    # only needed for plot_neighbors_atk
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    groundtruth_path = f"{cfg.labels}/{set_name}_attack_node_uuids.xz"
    log.info("reading ground truth from %s...", groundtruth_path)
    with lzma.open(groundtruth_path, "rt") as f:
        if cfg == config.atlasv1 or cfg == config.atlasv2:
            # use real UUIDs
            attack_node_uuids = {uuid.UUID(line.rstrip()) for line in f}
        else:
            attack_node_uuids = {line.rstrip() for line in f}
    attack_mask = make_attack_mask_uuids(g_node_ids, g, attack_node_uuids)

    #log.info("Plotting attack neighborhood labels...")
    #plot_neighbors_atk(cfg, labels, times, attack_mask,
    #                   plot_extra=shade_attack_periods,
    #                   out_name=f"{fullname}_labels_{n_hops}hop",
    #                   tz=tz)

    log.info("Plotting neighborhood labels...")
    plot_neighbors(cfg, labels, times, plot_extra=shade_attack_periods,
                   out_name=f"{fullname}_labels_{n_hops}hop",
                   tz=tz,
                   needs_brokenaxes=needs_brokenaxes)

    log.info("Plotting neighborhood label histogram...")
    hist_neighbors(cfg, labels, times, plot_extra=shade_attack_periods,
                   out_name=f"{fullname}_hist_{n_hops}hop",
                   tz=tz,
                   needs_brokenaxes=needs_brokenaxes)

def write_set_neighbors(cfg, set_name: str, n_hops: int = 2) -> None:
    labels_path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop.pkl.xz"
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"

    log.info("reading labels from %s...", labels_path)
    labels, _, g_node_ids, nbrs = util.pickle_read(labels_path)

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    if cfg == config.streamspot:
        g = g.g

    #graph_transforms.reverse_dns.rename_sockets(cfg, set_name, g)
    #graph_transforms.ip_whois.convert_sockets(g)

    out_path = f"{set_name}_nbrs_{n_hops}hop.txt"
    log.info("formatting neighborhoods to %s...", out_path)
    write_neighbors_txt(labels, nbrs, g_node_ids, g, n_hops, out_path)

def write_set_procs(cfg, set_name, n_hops: int = 1):
    labels_path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop.pkl.xz"
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"

    log.info("reading labels from %s...", labels_path)
    _, _, g_node_ids, _ = util.pickle_read(labels_path)

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    if cfg == config.streamspot:
        g = g.g

    out_path = f"{set_name}_procs_{n_hops}hop.txt"
    log.info("formatting process counts to %s...", out_path)
    write_procs_txt(g_node_ids, g, out_path)

def main(argv):
    for n_hops in [1, 2]:
        for set_name in ["cadets", "theia", "fivedirections", "trace"]:
            pickle_neighborhoods(config.e3, set_name, n_hops=n_hops, collapsed=True)
            plot_set(config.e3, set_name, n_hops=n_hops)
            #write_set_neighbors(config.e3, set_name, n_hops=n_hops)
            #write_set_procs(config.e3, set_name, n_hops=n_hops)

        for set_name in ["host201", "host501"]:
            pickle_neighborhoods(config.optc, set_name, n_hops=n_hops, collapsed=True)
            plot_set(config.optc, set_name, n_hops=n_hops)
            #write_set_neighbors(config.optc, set_name, n_hops=n_hops)
            #write_set_procs(config.optc, set_name, n_hops=n_hops)

        for set_name in ["h1", "h2"]:
            #set_name = f"{set_name}_benign"
            pickle_neighborhoods(config.atlasv2, set_name, n_hops=n_hops, collapsed=True)
            plot_set(config.atlasv2, set_name, n_hops=n_hops)
            #write_set_neighbors(config.atlasv2, set_name, n_hops=n_hops)
            #write_set_procs(config.atlasv2, set_name, n_hops=n_hops)

        #for set_name in ["h1", "h2"]:
        #    pickle_neighborhoods(config.atlasv1, set_name, n_hops=n_hops)
        #    plot_set(config.atlasv1, set_name, n_hops=n_hops, needs_brokenaxes=True)
        #    #write_set_neighbors(config.atlasv1, set_name, n_hops=n_hops)
        #    #write_set_procs(config.atlasv1, set_name, n_hops=n_hops)

        #for set_name in ["all"]:
        #    pickle_neighborhoods(config.streamspot, set_name, n_hops=n_hops)
        #    plot_set(config.streamspot, set_name, n_hops=n_hops)
        #    #write_set_neighbors(config.streamspot, set_name, n_hops=n_hops)
        #    #write_set_procs(config.streamspot, set_name, n_hops=n_hops)

        for set_name in ["trace"]:
            set_name = f"{set_name}_departitioned"
            pickle_neighborhoods(config.e3, set_name, n_hops=n_hops, collapsed=True)
            #plot_set(config.e3, set_name, n_hops=n_hops)
            #write_set_neighbors(config.e3, set_name, n_hops=n_hops)
            #write_set_procs(config.e3, set_name, n_hops=n_hops)
