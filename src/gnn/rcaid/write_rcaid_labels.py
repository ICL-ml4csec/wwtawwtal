"""Write node labels into RCAID format: a set of "{path}!~{id}" strings,
written as a PyTorch object."""

import logging
import lzma
from typing import Set
import uuid

import igraph as ig
import torch

import config
import gnn.util
import util
from graph_transforms.merge_sockets import collapse_sockets

log = logging.getLogger(__name__)

def fmt_label(u: ig.Vertex) -> str:
    """Return the RCAID-formatted node label of `u`"""
    return f"{u['name']}!~{u['nid']}"

def write_labels(out_path: str, g: ig.Graph, node_uuids: Set[str]) -> None:
    labels = set()
    for u in g.vs(uuid_in=node_uuids):
        labels.add(fmt_label(u))

    with lzma.open(out_path, "wb") as f:
        torch.save(labels, f)

def write_set_labels(cfg, set_name: str) -> None:
    #graph_path = f"{cfg.graphs}/{set_name}_prov_graph_relabeled_unknowns_collapsed_sockets.pkl.xz"
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    if cfg == config.streamspot:
        g = g.g

    g.vs["nid"] = list(range(g.vcount()))
    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    def read_uuids(cfg, path: str) -> set:
        with lzma.open(path, "rt") as f:
            if cfg == config.atlasv2:
                # need to transform to uuid.UUID
                uuids = {uuid.UUID(line.rstrip()) for line in f}
            else:
                uuids = {line.rstrip() for line in f}
        return uuids


    node_label_path = f"{cfg.labels}/{set_name}_attack_node_uuids.xz"
    log.info("reading attack node UUIDs from %s...", node_label_path)
    node_labels = read_uuids(cfg, node_label_path)

    # technically, directly using collapsed_sockets might mislabel sockets
    # but GNN only does classification on processes, so it won't matter
    #out_path = f"{cfg.rcaid_graphs_csv}/{set_name}_relabeled_unknowns_node_labels.pt.xz"
    out_path = f"{cfg.rcaid_graphs_csv}/{set_name}_node_labels.pt.xz"
    #out_path = f"{cfg.rcaid_graphs_csv}/{set_name}_unreduced_node_labels.pt.xz"
    log.info("writing RCAID-formatted node labels to %s...", out_path)
    write_labels(out_path, g, node_labels)

    contaminated_path = f"{cfg.labels}/{set_name}_contaminated_node_uuids.xz"
    try:
        log.info("reading contaminated node UUIDs from %s...", contaminated_path)
        contaminated_nodes = read_uuids(cfg, contaminated_path)

        out_path = f"{cfg.rcaid_graphs_csv}/{set_name}_node_contaminated.pt.xz"
        #out_path = f"{cfg.rcaid_graphs_csv}/{set_name}_unreduced_node_contaminated.pt.xz"
        log.info("writing RCAID-formatted contaminated nodes to %s...", out_path)
        write_labels(out_path, g, contaminated_nodes)
    except IOError:
        log.info("no contaminated nodes found, skipping...")

def write_streamspot_labels(cfg, set_name: str) -> None:
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    if cfg == config.streamspot:
        g = g.g

    # need to map non-collapsed node IDs to the collapsed ones
    log.info("mapping nids after merging sockets...")
    g.vs["nid"] = list(range(g.vcount()))

    collapse_sockets(g)
    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    attack_id_path = f"{cfg.labels}/{set_name}_nodes.xz"
    log.info("reading attack node IDs from %s...", attack_id_path)
    with lzma.open(attack_id_path, "rt") as f:
        attack_ids = {int(line.rstrip()) for line in f}

    out_path = f"{cfg.rcaid_graphs_csv}/{set_name}_node_labels.pt.xz"
    log.info("writing RCAID-formatted node labels to %s...", out_path)
    labels = set()
    for u in g.vs:
        if u["nid"] in attack_ids:
            labels.add(fmt_label(u))

    with lzma.open(out_path, "wb") as f:
        torch.save(labels, f)

def main(argv):
    #for set_name in ["cadets", "theia", "fivedirections", "trace"]:
    #for set_name in ["trace"]:
    #    write_set_labels(config.e3, set_name)
    #write_set_labels(config.e3, "trace_departitioned")
    write_streamspot_labels(config.streamspot, "all")
    #for set_name in ("h1", "h2"):
    #    write_set_labels(config.atlasv2, set_name)
    #write_set_labels(config.atlasv2, "combined")
