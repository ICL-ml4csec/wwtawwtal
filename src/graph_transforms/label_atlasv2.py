"""Produce node labels for ATLASv2 using legacy event labels."""

import logging
import lzma
import uuid
from typing import Set

import igraph as ig

import config
import util

log = logging.getLogger(__name__)

def label_nodes(set_name: str, g: ig.Graph,
                attack_event_uuids: Set[uuid.UUID]):
    log.info("collecting attack nodes...")
    attack_node_uuids = set()
    for e in g.es(uuid_in=attack_event_uuids):
        attack_node_uuids.add(g.vs[e.source]["uuid"])
        attack_node_uuids.add(g.vs[e.target]["uuid"])

    node_labels_path = f"{config.atlasv2.labels}/{set_name}_attack_node_uuids.xz"
    log.info("writing node labels to %s...", node_labels_path)
    with lzma.open(node_labels_path, "wt") as f:
        for uid in attack_node_uuids:
            f.write(str(uid))
            f.write("\n")

def label_set(set_name: str):
    graph_path = f"{config.atlasv2.graphs}/{set_name}_prov_graph.pkl.xz"
    edge_labels_path = f"{config.atlasv2.labels}/{set_name}_attack_event_uuids.xz"

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    log.info("reading labels from %s...", edge_labels_path)
    with lzma.open(edge_labels_path, "rt") as f:
        attack_event_uuids = {uuid.UUID(line.rstrip()) for line in f}

    label_nodes(set_name, g, attack_event_uuids)

def main(argv):
    for set_name in ["h1", "h2"]:
        label_set(set_name)
