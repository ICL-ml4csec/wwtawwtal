"""Output the "test-only" neighborhood as contaminated nodes for Cadets.

NOTE: this does correctly identify the neighborhoods by nid. However, because
of UUID non-uniqueness in the E3 datasets (especially based on how execs are
currently being parsed), the UUID contaminated set is approximately doubled.
We cannot fix this without switching to a better identifier scheme. In the
future, it would be better to generate our own UUIDs. Possible inputs are a
set of unique attributes per node (e.g., timestamp + name, may need more) or
just using a counter. Downside of the counter is that it's fragile to changes
in the parser, but upside is that it is more precise."""

import logging
import lzma

import config
import util

log = logging.getLogger(__name__)

def write_contaminated(n_hops: int = 1):
    graph_path = f"{config.e3.graphs}/cadets_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    labels_path = f"{config.e3.neighborhoods}/cadets_labels_{n_hops}hop.pkl.xz"
    log.info("reading neighborhoods from %s...", labels_path)
    labels, times, g_node_ids, _ = util.pickle_read(labels_path)

    # current neighborhood heuristics have the bad neighborhood at 369
    # NOTE: this may change depending on the graph format, iteration order,
    # number of hops, etc.
    bad_label = 369

    out_set = set()
    for uuid in g.vs(g_node_ids[labels == bad_label])["uuid"]:
        out_set.add(uuid)

    out_path = f"{config.e3.labels}/cadets_bad_nbr_node_uuids.xz"
    log.info("writing contaminated uuids to %s...", out_path)
    with lzma.open(out_path, "wt") as f:
        for uuid in out_set:
            f.write(uuid)
            f.write("\n")

def main(argv):
    write_contaminated()
