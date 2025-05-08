"""Write our igraph to a format usable by KAIROS.

We create a generalized KAIROS model with a single API, rather than having
separate definitions per dataset as in the provided code. Additionally, KAIROS
used a PostgreSQL database in its original implementation with some finnicky
libraries, but the database is only used for graph construction, and doesn't
appear to be needed for the actual model training/testing code, which is what
we're interested in. Therefore, we use a CSV-based plaintext format (instead
of sqlite3, which was another possible consideration) to pass the data.

KAIROS works on streaming edges; the input graph is a PyG TemporalData object,
which contains a list of edges (srcid, dstid, time, msg), where each attribute
is actually stored in a separate torch tensor. `srcid` and `dstid` are numeric
node IDs, presumably increasing from 0. `time` is the timestamp in seconds,
and `msg` is an embedding of the event as a float. The embedding itself is
generated using other metadata, so we need to pass the raw graph to KAIROS and
let it embed the data. They use a similar NODE EDGE NODE embedding scheme.

The input format is:

nodes.csv:
    uuid, type, name

    The name and type are needed for the hierarchical hashing-based node
    embedding. nid and UUID is here for our own use later (e.g., to retrofit
    labels). The nodes are also in the same order as the graph.

edges.csv:
    srcid, dstid, type, time

    The `srcid` and `dstid` indices must correlate to the (implicit) node
    indices in `nodes.csv`. Additionally, the edges should be sorted in
    ascending timestamp order.

config.json:
    train_end: index of the last training edge

    Much like FLASH, KAIROS doesn't do early stopping with a validation set,
    and instead hardcodes the number of epochs. We'll just recycle the same
    epoch count, as KAIROS appears to have been trained against entire
    datasets, so the graph size should be comparable.

There is additionally true_nid.csv, which is essentially just another column
for nodes.csv that contains the true graph nid. Since UUID is unreliable for
the DARPA datasets, we need a way to identify nodes with no ambiguity. (This
should have probably been in the original nodes.csv, but to avoid unneeded
work we just add it later.)
"""

import csv
import json
import logging
import lzma

import igraph as ig
import tqdm

import config
import gnn.util.benign_day_edges
import graph_transforms.sample_edges
import util

log = logging.getLogger(__name__)

def node_name(u: ig.Vertex) -> str:
    # these should have been removed by `transform_graph`, but check anyways
    if u["name"].startswith("\x00"):
        # replace null bytes for unknowns with "?", since nulls confuse pandas
        return (u["name"].replace("\x00", "?"),)
    if u["type"] == "socket":
        name = u["name"]
        if u["name"].startswith("sock="):
            name = u["name"][5:]
        return name
    return u["name"]

def to_kairos(cfg, set_name: str, g: ig.Graph) -> None:
    # find last training edge
    if cfg == config.e3:
        split_name = set_name
        if split_name == "trace_departitioned":
            split_name = "trace"
        t_end = gnn.util.benign_day_edges.e3_time_train_end(split_name)
    elif cfg == config.atlasv2:
        t_end = gnn.util.benign_day_edges.atlasv2_time_train_end()
    elif cfg == config.streamspot:
        pg = g
        g = pg.g
        _, t_end = gnn.util.benign_day_edges.streamspot_time_train_end(pg)
    elif cfg == config.optc:
        t_end = gnn.util.benign_day_edges.optc_time_train_end(set_name)
    else:
        raise ValueError("no train/test split available")

    edges_path = f"{cfg.kairos}/{set_name}_edges.csv.xz"
    log.info("writing edges to %s...", edges_path)
    with lzma.open(edges_path, "wt", newline="") as f:
        fieldnames = ("srcid", "dstid", "type", "time")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # force iteration in time order
        for i, e in enumerate(tqdm.tqdm(sorted(g.es, key=lambda e: e["time"]))):
            row = {
                "srcid": e.source,
                "dstid": e.target,
                "type": e["type"],
                "time": int(e["time"]), # streamspot: force int
            }
            writer.writerow(row)
            if e["time"] > t_end:
                train_end = i
                t_end = float("inf") # prevent condition from being True

    nodes_path = f"{cfg.kairos}/{set_name}_nodes.csv.xz"
    log.info("writing nodes to %s...", nodes_path)
    with lzma.open(nodes_path, "wt", newline="") as f:
        fieldnames = ("uuid", "type", "name")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(range(len(g.vs))):
            u = g.vs[i]
            row = {
                "uuid": u["uuid"],
                "type": u["type"],
                "name": node_name(u),
            }
            writer.writerow(row)

    config_path = f"{cfg.kairos}/{set_name}_config.json"
    log.info("writing config to %s...", config_path)
    with open(config_path, "wt") as f:
        json.dump({
            "train_end": train_end
        }, f)

def write_true_nid(cfg, set_name, g: ig.Graph):
    """Write true nids, and also transform the graph in-place."""
    # add nid attribute
    g.vs["nid"] = list(range(g.vcount()))

    gnn.util.transform_graph(g, set_name, remove_parallel=False, filter_edges=False)

    out_path = f"{cfg.kairos}/{set_name}_true_nids.csv.xz"
    log.info("writing true nids to %s...", out_path)
    with lzma.open(out_path, "wt", newline="") as f:
        fieldnames = ("nid",)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(range(len(g.vs))):
            u = g.vs[i]
            row = {
                "nid": u["nid"],
            }
            writer.writerow(row)

    return g

def main(argv):
    #cfg = config.e3
    #for set_name in ["cadets", "theia", "fivedirections", "trace", "trace_departitioned"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    g = write_true_nid(cfg, set_name, g)
    #    #to_kairos(cfg, set_name, g)

    #cfg = config.atlasv2
    #for set_name in ["h1", "h2"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    g = write_true_nid(cfg, set_name, g)
    #    to_kairos(cfg, set_name, g)

    #cfg = config.streamspot
    #for set_name in ["all"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    pg = util.pickle_read(graph_path)
    #    pg.g = g
    #    g = write_true_nid(cfg, set_name, g)
    #    #to_kairos(cfg, set_name, pg)

    #cfg = config.optc
    #for set_name in ["host201", "host501"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    g = write_true_nid(cfg, set_name, g)
    #    to_kairos(cfg, set_name, g)

    #cfg = config.e3
    #for set_name in ["trace", "trace_departitioned"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    g = write_true_nid(cfg, set_name, g)
    #    to_kairos(cfg, set_name, g)

    cfg = config.e3
    for set_name in ["trace"]:
        graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
        log.info("reading %s...", graph_path)
        g = util.pickle_read(graph_path)
        # limit to at most ~1e6 edges per node
        log.info("base graph:")
        log.info("  |V| = %s, |E| = %s", f"{g.vcount():,}", f"{g.ecount():,}")
        graph_transforms.sample_edges.sample_edges(g, threshold=int(1e6))
        log.info("post-sample:")
        log.info("  |V| = %s, |E| = %s", f"{g.vcount():,}", f"{g.ecount():,}")
        g = write_true_nid(cfg, set_name, g)
        to_kairos(cfg, set_name, g)
