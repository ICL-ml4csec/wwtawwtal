"""Count the number of edges in known benign days of datasets. WARNING: these
functions will simplify the input graphs."""

import datetime
import logging

import igraph as ig

import config
import util
import gnn.util

log = logging.getLogger(__name__)

def e3_time_train_end(set_name: str) -> int:
    """Return the timestamp of the last day of training for E3."""
    if set_name == "cadets":
        day = 6
    elif set_name == "theia":
        day = 10
    elif set_name == "fivedirections":
        day = 9 # true start of attack period
        #day = 6 # reduced set
    elif set_name.startswith("trace"):
        day = 10 # true start of attack period
        #day = 6 # reduced set
    end = int(1e9 * datetime.datetime(2018, 4, day, tzinfo=util.EDT).timestamp())
    return end

def count_tc_edges(g: ig.Graph, set_name: str):
    end = e3_time_train_end(set_name)
    return end, len(g.es(time_lt=end))

def _count_tc_edges(g: ig.Graph, set_name: str):
    _, n_benign = count_tc_edges(g, set_name)
    log.info("total edges: %s", g.ecount())
    log.info("%s: number of benign edges: %s", set_name, n_benign)

def atlasv2_time_train_end() -> int:
    end = int(1e9 * datetime.datetime(2022, 7, 18, tzinfo=util.CDT).timestamp())
    return end

def count_atlasv2_edges(g: ig.Graph):
    end = atlasv2_time_train_end()
    return end, len(g.es(time_lt=end))

def _count_atlasv2_edges(g: ig.Graph, set_name: str):
    _, n_benign = count_atlasv2_edges(g)
    log.info("total edges: %s", g.ecount())
    log.info("%s: number of benign edges: %s", set_name, n_benign)

def _count_tc(set_name: str):
    #graph_path = f"{config.e3.graphs}/{set_name}_prov_graph_relabeled_unknowns.pkl.xz"
    graph_path = f"{config.e3.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #graph_path = f"{config.e3.graphs}/{set_name}_prov_graph_relabeled_unknowns_collapsed_sockets.pkl.xz"

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    _count_tc_edges(g, set_name)

def _count_atlasv2(set_name: str):
    #graph_path = f"{config.atlasv2.graphs}/{set_name}_prov_graph_relabeled_unknowns_collapsed_sockets.pkl.xz"
    graph_path = f"{config.atlasv2.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    _count_atlasv2_edges(g, set_name)

def streamspot_time_train_end(pg: "ProvenanceGraph"):
    train_end = pg.trial_boundaries[360]
    val_end = pg.trial_boundaries[400]

    return train_end, val_end

def count_streamspot_edges(pg: "ProvenanceGraph", g: ig.Graph):
    train_end, val_end = streamspot_time_end(pg)

    n_train = len(g.es(time_lt=train_end))
    n_val = len(g.es(time_lt=val_end))

    return train_end, val_end, n_train, n_val

def _count_streamspot(set_name: str):
    pgraph_path = f"{config.streamspot.graphs}/{set_name}_prov_graph.pkl.xz"
    graph_path = f"{config.streamspot.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"

    log.info("reading ProvenanceGraph from %s...", pgraph_path)
    pg = util.pickle_read(pgraph_path)

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    _, _, train_end, val_end = count_streamspot_edges(pg, g)

    log.info("total edges: %s", g.ecount())
    log.info("%s: end of train: %s", set_name, train_end)
    log.info("%s: end of val: %s", set_name, val_end)

def optc_time_train_end(set_name: str) -> int:
    """Return the timestamp of the last day of training for OpTC."""
    # since first attack is generally quite late,
    # leave at least a full benign day for test
    if set_name == "host201":
        day = 23
    elif set_name == "host501":
        day = 23
    else:
        raise ValueError(f"Undefined split for {set_name}")
    end = int(1e9 * datetime.datetime(2019, 9, day, tzinfo=util.EDT).timestamp())
    return end

def count_optc_edges(g: ig.Graph, set_name: str):
    end = optc_time_train_end(set_name)
    return end, len(g.es(time_lt=end))

def _count_optc_edges(g: ig.Graph, set_name: str):
    _, n_benign = count_optc_edges(g, set_name)
    log.info("total edges: %s", g.ecount())
    log.info("%s: number of benign edges: %s", set_name, n_benign)

def _count_optc(set_name: str):
    graph_path = f"{config.optc.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    _count_optc_edges(g, set_name)

def main(argv):
    #for set_name in ("cadets", "theia", "fivedirections", "trace"):
    #for set_name in ["theia"]:
    #    _count_tc(set_name)
    #_count_tc("trace_departitioned")
    #for set_name in ("h1", "h2"):
    #    _count_atlasv2(set_name)
    #_count_atlasv2("combined")
    #_count_streamspot("all")
    for set_name in ["host201", "host501"]:
        _count_optc(set_name)
