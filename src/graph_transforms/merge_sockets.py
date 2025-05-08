"""Certain graphs have multiple socket nodes with identical addresses. This
merges the identical sockets into a single node."""

import logging

import igraph as ig
import numpy as np

import config
import util

log = logging.getLogger(__name__)

def collapse_sockets(g: ig.Graph):
    skts = g.vs(type_eq="socket")
    id_mapping = np.arange(g.vcount())
    skt_map = {}
    to_delete = []
    for s in skts:
        if s["name"] not in skt_map:
            skt_map[s["name"]] = s.index
        else:
            id_mapping[s.index] = skt_map[s["name"]]
            to_delete.append(s.index)
    g.contract_vertices(id_mapping, combine_attrs="first")

    # really weird quirk: if the last N vertices are all contracted, they are
    # automatically deleted. just remove any indices larger than the current
    # graph size
    to_delete = [i for i in to_delete if i < g.vcount()]

    # contract won't delete the old nodes
    g.delete_vertices(to_delete)

def collapse_graph(cfg, graph_name: str):
    graph_path = f"{cfg.graphs}/{graph_name}.pkl.xz"
    out_path = f"{cfg.graphs}/{graph_name}_collapsed_sockets.pkl.xz"

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)
    if cfg == config.streamspot:
        g = g.g

    log.info("pre-merge size: |V| = %s, |E| = %s", g.vcount(), g.ecount())

    log.info("collapsing sockets...")
    collapse_sockets(g)

    log.info("post-merge size: |V| = %s, |E| = %s", g.vcount(), g.ecount())

    log.info("writing result to %s...", out_path)
    util.pickle_write(g, out_path)

def main(argv):
    #for set_name in ["cadets", "theia", "fivedirections", "trace"]:
    #    collapse_graph(config.e3, f"{set_name}_prov_graph")
    #collapse_graph(config.e3, "trace_departitioned_prov_graph")
    #for set_name in ("h1", "h2"):
    #    collapse_graph(config.atlasv2, f"{set_name}_prov_graph")
    #collapse_graph(config.streamspot, f"all_prov_graph")
    for set_name in["host201", "host501"]:
        collapse_graph(config.optc, f"{set_name}_prov_graph")
