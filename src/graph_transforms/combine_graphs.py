"""Produce disjoint unions of graphs; i.e., this just literally puts two
separate graphs into the same igraph object."""

import logging

import igraph as ig

import config
import util

log = logging.getLogger(__name__)

def combine_graphs(g1: ig.Graph, g2: ig.Graph) -> ig.Graph:
    return ig.disjoint_union([g1, g2])

def combine_atlasv2(suffix: str = "") -> None:
    h1_path = f"{config.atlasv2.graphs}/h1_prov_graph{suffix}.pkl.xz"
    h2_path = f"{config.atlasv2.graphs}/h2_prov_graph{suffix}.pkl.xz"
    out_path = f"{config.atlasv2.graphs}/combined_prov_graph{suffix}.pkl.xz"

    log.info("loading graph from %s...", h1_path)
    h1 = util.pickle_read(h1_path)

    log.info("loading graph from %s...", h2_path)
    h2 = util.pickle_read(h2_path)

    log.info("taking the disjoint union...")
    g = combine_graphs(h1, h2)

    log.info("writing output to %s...", out_path)
    util.pickle_write(g, out_path)

def main(argv):
    combine_atlasv2("_collapsed_sockets")
