"""Format our particular provenance graphs to an output usable by FLASH.

Our provenance graphs do not necessarily retain all of the information wanted
by flash (e.g., command lines, etc.), but we can still output as much as we
have. We must output both the node neighborhoods for word2vec embeddings and
the provenance graph itself. For every node, every associated edge is embedded
using word2vec by phrases constructed via "{NODE ATTRS} + {EDGE ATTRS} + {NODE
ATTRS}" in time order; the mean of these embeddings is the node embedding.

The graph format just uses torch-geometric's `Data`; rather than constructing
the Data now, we'll just output a format convenient for it. The final graph
needs node embeddings, node labels (in the sense of node types, not anomaly
labels), and edge connectivity. Graph node ids (n_id) are automatically
inferred based on the order of the input nodes. We want to allow FLASH to
control the word2vec embedding model, so we need to additionally output the
phrases associated with each node and the list of all training phrases.

We use a generalized FLASH model (as opposed to the per-dataset provided
notebooks) so that we can specify one overall API here. We will output these
to the following files:

nodes.csv: The graph nodes in CSV form, formatted as:
    uuid, type, edge_indices

    uuid is self-explanatory, and helpful for correlating nodes later.

    type: node type, used as GNN training labels. Our set of node types (only
    file, process, socket, and registry keys) is slightly smaller than the
    original FLASH set, which includes module (i.e., libraries) for OpTC and
    UNIX sockets, unnamed pipes, and directories for E3.

    edge_indices: a list of integer indices that indicate which edges in
    edges.csv correspond to this node, __in time order__. Note that this edge
    index is NOT the same as the edge id, since the edges are also sorted by
    timestamp.

    Note that node id is implicitly being tracked as the row number.

edges.csv: Graph edges in CSV form sorted by timestamp, formatted in COO form
    and including "phrases" used for word2vec embeddings:

    edge_id, src_index, dst_index, phrase

    edge_id is the edge index in the original graph. This is only here in
    case edges need to be correlated with the original graph edges.

    `phrase` is a list of tokens (i.e., strings). The phrase format is
    flexible due to the use of word2vec. We slightly break from the FLASH
    phrase formatting, which is dependent on both the node and edge types, and
    instead use a generic NODE - EDGE - NODE format, specifically:

    {src_path} {edge_type} {dst_path}

    We are notably missing command lines (since we didn't retain them in the
    graph), but actually have almost all of the remaining expected data. The
    one additional postprocessing step is to break apart socket names into the
    pair (ip port), instead of keeping them as one value. Additionally, note
    that full path names are used instead of just filenames.

    Another departure from FLASH is that we encode directionality here via the
    order of {src_path} and {dst_path}, while FLASH either adds it as an
    explicit parameter (e.g., in OpTC "FLOW" events) or as an implicit
    parameter tied to the edge type (i.e., the GNN would have to learn that
    embeddings with "read" are different directions than "write").

config.json: Configuration info for this dataset, e.g., the train/test split.

    train_end: index of the last training edge

    Note that FLASH by default doesn't use a validation set, instead just
    hardcoding the number of epochs. We use this same methodology because
    trying to retrofit a validation set would be difficult, as the
    connectivity of the graph is important in GNNs, and the connectivitiy
    itself is sensitive to time slicing.

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
from typing import Tuple

import igraph as ig
import tqdm

import config
import gnn.util
import gnn.util.benign_day_edges
import graph_transforms.sample_edges
import util

log = logging.getLogger(__name__)

def node_name(g: ig.Graph, nid: int) -> Tuple[str]:
    u = g.vs[nid]
    # these should have been removed by `transform_graph`, but check anyways
    if u["name"].startswith("\x00"):
        # replace null bytes for unknowns with "?", since nulls confuse pandas
        return (u["name"].replace("\x00", "?"),)
    if u["type"] == "socket":
        name = u["name"]
        if u["name"].startswith("sock="):
            name = u["name"][5:]
        # careful with IPv6. Based on our formatting, this should always work
        # (unless it's a UNIX domain socket, then treat as a file)
        try:
            ip, port = name.rsplit(":", maxsplit=1)
        except ValueError:
            return (u["name"],)
        return (ip, port)
    return (u["name"],)

def to_flash(cfg, set_name: str, g: ig.Graph) -> None:
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

    ## TEMPORARY: use same CADETS time split as FLASH
    #t_start = 1522828474810632615
    #t_end = 1522949717817921260
    #test_start = 1523478987898404334
    #test_end = 1523586247675522567

    ## remove all edges that aren't in these time ranges
    #g.es(time_lt=t_start).delete()
    #g.es(time_gt=test_end).delete()
    #g.es(time_gt=t_end, time_lt=test_start).delete()
    ## ...now remove all the 0-degree vertices
    #g.vs(_degree=0).delete()

    edges_path = f"{cfg.flash}/{set_name}_edges.csv.xz"
    log.info("writing edges to %s...", edges_path)
    edge_map = {} # edge id -> sorted csv edge index
    with lzma.open(edges_path, "wt", newline="") as f:
        fieldnames = ("edge_id", "src_index", "dst_index", "phrase")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # force iteration in time order
        for i, e in enumerate(tqdm.tqdm(sorted(g.es, key=lambda e: e["time"]))):
            phrase = [
                *node_name(g, e.source),
                e["type"],
                *node_name(g, e.target),
            ]
            row = {
                "edge_id": e.index,
                "src_index": e.source,
                "dst_index": e.target,
                "phrase": phrase,
            }
            writer.writerow(row)
            if e["time"] > t_end:
                train_end = i
                t_end = float("inf") # prevent condition from being True
            edge_map[e.index] = i

    nodes_path = f"{cfg.flash}/{set_name}_nodes.csv.xz"
    log.info("writing nodes to %s...", nodes_path)
    with lzma.open(nodes_path, "wt", newline="") as f:
        fieldnames = ("uuid", "type", "edge_indices")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(range(len(g.vs))):
            # explicit iteration in order; technically unnecessary
            u = g.vs[i]
            edges = sorted(g.es(_incident=(u.index,)), key=lambda e: e["time"])
            row = {
                "uuid": u["uuid"],
                "type": u["type"],
                "edge_indices": [edge_map[e.index] for e in edges],
            }
            writer.writerow(row)

    config_path = f"{cfg.flash}/{set_name}_config.json"
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

    out_path = f"{cfg.flash}/{set_name}_true_nids.csv.xz"
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
    # NOTE: trace is crashing, is probably related to the millions of
    # duplicate edges. Might need to do something about that (limit edges?)
    #cfg = config.e3
    #for set_name in ["cadets", "theia", "fivedirections", "trace", "trace_departitioned"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    g = write_true_nid(cfg, set_name, g)
    #    gnn.util.transform_graph(g, set_name, remove_parallel=False, filter_edges=False)
    #    #to_flash(cfg, set_name, g)

    #cfg = config.atlasv2
    #for set_name in ["h1", "h2"]:
    #    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #    log.info("reading %s...", graph_path)
    #    g = util.pickle_read(graph_path)
    #    g = write_true_nid(cfg, set_name, g)
    #    #to_flash(cfg, set_name, g)

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
    #    #to_flash(cfg, set_name, pg)

    cfg = config.optc
    for set_name in ["host201", "host501"]:
        graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
        log.info("reading %s...", graph_path)
        g = util.pickle_read(graph_path)
        g = write_true_nid(cfg, set_name, g)
        gnn.util.transform_graph(g, set_name, remove_parallel=False, filter_edges=False)
        to_flash(cfg, set_name, g)


    cfg = config.e3
    for set_name in ["trace", "trace_departitioned"]:
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
        to_flash(cfg, set_name, g)
