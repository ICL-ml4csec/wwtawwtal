"""Undo the event partitioning in TRACE by merging all children of EVENT_UNIT
calls with the parent process."""

import logging
import lzma
from typing import Dict

import igraph as ig
import numpy as np
import tqdm

import config
import util

log = logging.getLogger(__name__)

def departition_units(g: ig.Graph):
    """Departition `g`, and return a mapping of UUIDs to their merged node's
    UUID."""
    units = g.es(type_eq="EVENT_UNIT")
    id_mapping = np.arange(g.vcount())
    uuid_mapping = {}
    to_delete = []
    # EVENT_UNIT is process -> unit
    log.info("finding event units...")
    for u in tqdm.tqdm(units, total=len(units)):
        id_mapping[u.target] = u.source
        uuid_mapping[g.vs[u.target]["uuid"]] = g.vs[u.source]["uuid"]
        to_delete.append(u.target)

    log.info("merging units and their parents...")
    g.contract_vertices(id_mapping, combine_attrs="first")

    log.info("removing obselete nodes...")
    g.delete_vertices(to_delete)

    return uuid_mapping

def write_departitioned_labels(cfg, set_name: str, uuid_mapping: Dict):
    attack_path = f"{cfg.labels}/{set_name}_attack_node_uuids.xz"
    log.info("reading attack labels from %s...", attack_path)
    with lzma.open(attack_path, "rt") as f:
        attack_uuids = {line.rstrip() for line in f}

    # remap merged attack UUIDs
    attack_uuids = {uuid_mapping.get(uuid, uuid) for uuid in attack_uuids}

    attack_out_path = f"{cfg.labels}/{set_name}_departitioned_attack_node_uuids.xz"
    log.info("writing fixed labels to %s...", attack_out_path)
    with lzma.open(attack_out_path, "wt") as f:
        for uuid in attack_uuids:
            f.write(f"{uuid}\n")

    contam_path = f"{cfg.labels}/{set_name}_contaminated_node_uuids.xz"
    log.info("reading contaminated labels from %s...", contam_path)
    with lzma.open(contam_path, "rt") as f:
        contam_uuids = {line.rstrip() for line in f}

    # remap merged contaminated UUIDs
    contam_uuids = {uuid_mapping.get(uuid, uuid) for uuid in contam_uuids}

    contam_out_path = f"{cfg.labels}/{set_name}_departitioned_contaminated_node_uuids.xz"
    log.info("writing fixed labels to %s...", contam_out_path)
    with lzma.open(contam_out_path, "wt") as f:
        for uuid in contam_uuids:
            # if both attack and contaminated, make attack the priority
            if uuid not in attack_uuids:
                f.write(f"{uuid}\n")

def departition_graph(cfg, set_name: str):
    graph_name = f"{set_name}_prov_graph"
    graph_path = f"{cfg.graphs}/{graph_name}.pkl.xz"
    out_path = f"{cfg.graphs}/{set_name}_departitioned_prov_graph.pkl.xz"

    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    log.info("pre-merge size: |V| = %s, |E| = %s", g.vcount(), g.ecount())

    uuid_mapping = departition_units(g)
    write_departitioned_labels(cfg, set_name, uuid_mapping)
    del uuid_mapping

    #log.info("eliminating parallel edges...")
    #g.simplify(multiple=True, loops=False, combine_edges="first")

    log.info("post-merge size: |V| = %s, |E| = %s", g.vcount(), g.ecount())

    log.info("writing result to %s...", out_path)
    util.pickle_write(g, out_path)

def main(argv):
    departition_graph(config.e3, "trace")
