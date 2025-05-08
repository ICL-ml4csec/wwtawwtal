"""Convert streamspot nid label to UUID labeling."""

import logging
import lzma

import config
import util

log = logging.getLogger(__name__)

def write_streamspot_uuid_labels(set_name: str) -> None:
    graph_path = f"{config.streamspot.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    pg = util.pickle_read(graph_path)
    g = pg.g

    labels_path = f"{config.streamspot.labels}/{set_name}_nodes.xz"
    log.info("reading node nid labels from %s...", labels_path)
    with lzma.open(labels_path, "rt") as f:
        nids = [int(line.strip()) for line in f]

    out_path = f"{config.streamspot.labels}/{set_name}_attack_node_uuids.xz"
    with lzma.open(out_path, "wt") as f:
        for uuid in g.vs(nids)["uuid"]:
            f.write(uuid)
            f.write("\n")

def main(argv):
    write_streamspot_uuid_labels("all")
