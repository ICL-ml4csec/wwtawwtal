"""Given a particular attack subgraph, append all events related to malicious
processes, then write both the list of event and node UUIDs.

WARNING: E3's event UUIDs are not very unique at all; most event UUIDs refer
to at least 2 or more events, all of which may be totally unrelated. Node
UUIDs are probably a safer bet."""

import glob
import logging
import lzma
from typing import Dict, List, Tuple

import igraph as ig

import config
from parse.darpa_e3 import json_stream
import util

log = logging.getLogger(__name__)

# ideally we would rely on type=process, but there are also some unknown nodes
# that appear to be processes too. look for clear indicators of process
# behavior instead
def is_process(v: ig.Vertex) -> bool:
    if v["type"] == "process":
        return True
    elif v["type"] != "unknown":
        return False
    # look for incoming exec, clone in either direction
    if (len(v.graph.es(_target=v.index, type_eq="exec")) > 0
        or len(v.graph.es(_target=v.index, type_eq="clone")) > 0
        or len(v.graph.es(_source=v.index, type_eq="clone")) > 0):
        return True
    return False

def create_node_mapping(g: ig.Graph) -> Dict[str, int]:
    """Create a node UUID -> min timestamp mapping from subgraph `g`."""
    process_nodes = {} # UUID -> min time
    max_t = max(g.es["time"])
    for v in g.vs(is_process):
        def min_nonzero_time(es: ig.EdgeSeq):
            return min(filter(lambda t: t > 0, es["time"]), default=max_t)
        min_t = min(min_nonzero_time(g.es(_source=v.index)),
                    min_nonzero_time(g.es(_target=v.index)))
        process_nodes[v["uuid"]] = min_t
    return process_nodes

class ParseState(json_stream.AbstractParseState):
    def __init__(self, set_name: str, graphs: List[Tuple[str, ig.Graph]]):
        """Reparse TC data to add all events for attack subgraphs.
        `set_name`: dataset to be reparsed
        `graphs`: List of (subgraph name, graph) pairs"""
        super().__init__(set_name)
        self.subgraphs = [(name, create_node_mapping(g), [], [])
                          for name, g in graphs]

    def parse_event(self, event: Dict) -> None:
        # need to check both subject and object,
        # since we want both directions of events
        try:
            sub_uuid = event["subject"][f"{self.ns}UUID"]
        except (KeyError, TypeError):
            sub_uuid = None

        try:
            obj_uuid = event["predicateObject"][f"{self.ns}UUID"]
        except (KeyError, TypeError):
            obj_uuid = None

        for _, process_nodes, attack_node_uuids, attack_event_uuids in self.subgraphs:
            if sub_uuid in process_nodes:
                min_t = process_nodes[sub_uuid]
                if event["timestampNanos"] >= min_t:
                    attack_event_uuids.append(event["uuid"])
                    attack_node_uuids.append(sub_uuid)
                    if obj_uuid is not None:
                        attack_node_uuids.append(obj_uuid)
            elif obj_uuid in process_nodes:
                # if both are in process_nodes, subject only is fine anyways
                min_t = process_nodes[obj_uuid]
                if event["timestampNanos"] >= min_t:
                    attack_event_uuids.append(event["uuid"])
                    attack_node_uuids.append(obj_uuid)
                    if sub_uuid is not None:
                        attack_node_uuids.append(sub_uuid)

    def parse_record(self, record: Dict) -> None:
        try:
            for k, v in record["datum"].items():
                if k == f"{self.ns}Event":
                    self.parse_event(v)
                return
        except BaseException:
            log.error("Error at record: %s", record)
            raise

def reparse_events(subgraph_names: List[str], tar_dir: str) -> None:
    """Given attack subgraphs, reparse events to add non-provenance events,
    then write all event UUIDs to attack_event_uuids/`subgraph_name`."""
    set_name = util.tc_set_name(tar_dir)
    graphs = []
    for name in subgraph_names:
        subgraph_path = f"{config.e3.attack_subgraphs}/{name}"
        log.info("reading %s", subgraph_path)
        sg = util.pickle_read(subgraph_path)
        graphs.append((name, sg))

    log.info("collecting all attack-related events of %s", set_name)
    state = ParseState(set_name, graphs)
    json_stream.parse_dataset(tar_dir, state)

    for name, _, attack_node_uuids, attack_event_uuids in state.subgraphs:
        name = name.rsplit(".", maxsplit=2)[0]
        events_path = f"{config.e3.attack_event_uuids}/{name}.xz"
        log.info("writing event UUIDs to %s", events_path)
        with lzma.open(events_path, "wt") as f:
            for uuid in attack_event_uuids:
                f.write(uuid)
                f.write("\n")

        nodes_path = f"{config.e3.attack_node_uuids}/{name}.xz"
        log.info("writing node UUIDS to %s", nodes_path)
        with lzma.open(nodes_path, "wt") as f:
            for uuid in attack_node_uuids:
                f.write(uuid)
                f.write("\n")

def do_all_subgraphs() -> None:
    base = "<PATH_TO_DARPA_E3>"
    tar_dirs = {
        "cadets": f"{base}/cadets-complete",
        "fivedirections": f"{base}/fivedirections-complete",
        "theia": f"{base}/theia-complete",
        "trace": f"{base}/trace",
    }

    for set_name, tar_dir in tar_dirs.items():
        log.info("Reparsing all %s events...\n", set_name)
        paths = glob.glob(f"{config.e3.attack_subgraphs}/{set_name}*.pkl.xz")
        names = [p.rsplit("/", maxsplit=1)[1] for p in paths]
        reparse_events(names, tar_dir)

def main(argv):
    do_all_subgraphs()
