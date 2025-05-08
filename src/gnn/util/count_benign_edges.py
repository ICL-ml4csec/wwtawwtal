"""Count the total number of benign edges before the first attack edge is
encountered."""

import logging
import lzma
from typing import Set

import igraph as ig
import tqdm

import config
import gnn.util
import util

log = logging.getLogger(__name__)

def count_edges(g: ig.Graph, attack_edge_uuids: Set[str]) -> int:
    i = 0
    for e in tqdm.tqdm(sorted(g.es, key=lambda e: e["time"])):
        if e["uuid"] in attack_edge_uuids:
            return i
        i += 1

def report_count(cfg, set_name: str) -> None:
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    gnn.util.transform_graph(g, set_name)

    label_path = f"{cfg.labels}/{set_name}_attack_event_uuids.xz"
    log.info("reading labels from %s...", label_path)
    with lzma.open(label_path, "rt") as f:
        attack_edge_uuids = {line.rstrip() for line in f}

    n = count_edges(g, attack_edge_uuids)
    log.info("%s: edges before attack: %s", set_name, n)

def main(argv):
    for set_name in ("cadets", "theia", "fivedirections", "trace"):
        report_count(config.e3, set_name)
