"""Collect all events into a compact representation."""

import logging
import lzma
import sys
from typing import Dict, NamedTuple, Optional, TextIO
import uuid

import igraph as ig

import config
import parse.graph
from parse import events
from parse.atlasv2 import json_stream
import util

log = logging.getLogger(__name__)

class ParseState(json_stream.AbstractParseState):
    def __init__(self, host: str, edge_uuid_pname_map: Dict[str, str],
                 out_file: TextIO, labeldir: Optional[str] = None):
        """Create an event parser. Optionally, if given a `labeldir` that
        contains files for each attack with the EventRecordID of all attack
        events, add labels to the data.

        Note that `edge_uuid_pname_map` maps an *edge UUID* to its *process
        name* (not node UUIDs). This is because the ATLASv2 UUID generation
        scheme I chose will consistently give the same edge UUIDs, but the
        node UUID actually depends on which events are parsed and which are
        not, so the node UUIDs of the event stream cannot be assumed to
        correlate with the provenance graph. Mapping nodes would be preferable
        if possible, since the mapping would be smaller, but the ATLASv2
        dataset is also small enough that it should not matter too much.

        Despite this, the event stream does also output process UUIDs to
        distinguish which process output which event. As mentioned before,
        this cannot be correlated to the provenance graph UUID, but is still
        necessary, e.g., to avoid spurious windows in sequence analysis."""
        super().__init__(host)
        self.writer = events.EventsWriter(out_file)
        self.labeldir = labeldir
        self.edge_uuid_pname_map = edge_uuid_pname_map
        self.attack_event_uuids = set()
        self.attack_node_ids = set()
        self.pid_table = {}
        self.next_id = 0

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *_) -> None:
        self.writer.flush()

    def map_pid(self, pid: int) -> int:
        """Map `pid` to a node ID, and return the node ID."""
        if pid not in self.pid_table:
            self.pid_table[pid] = self.next_id
            self.next_id += 1
        return self.pid_table[pid]

    def remove_pid(self, pid: int) -> None:
        """Remove `pid` from the table. This means the next time `pid`
        appears, it will be considered a new node."""
        self.pid_table.pop(pid, None)

    def reset_state(self) -> None:
        """Reset mapping state. This is especially important for the attacks
        which were reverted from snapshots; the nodes will not correlate
        anymore.

        Calling this will result in multiple disjoint subgraphs."""
        # TODO better to smash the graphs together or separate?
        self.pid_table = {}

    def make_edge_uuid(self, event: NamedTuple) -> uuid.UUID:
        """Create a UUID for event.

        This isn't one of the standardized UUIDs, but it's guaranteed to be
        unique within the ATLASv2 dataset, including across hosts. This
        (ab)uses the Windows EventRecordId being a long (i.e., at most 64-bit)
        identifier which is unique within one event stream; by combining this
        with the dataset name squished into a 64-bit number, the result is
        guaranteed to be unique.
        """
        b = self.event_uuid_prefix + event.EventRecordID.to_bytes(8, "big")
        return uuid.UUID(bytes=b)

    def parse_record(self, record: NamedTuple) -> None:
        event = record.EventID
        time = record.SystemTime
        edge_uuid = self.make_edge_uuid(record)
        proc_name = self.edge_uuid_pname_map.get(edge_uuid,
                                                 parse.graph.unknown_proc)

        if record.EventID == 5156: # connect
            try:
                pid = int(record.EventDataFields["ProcessId"], 10)
            except KeyError: # ATLASv2 connect
                pid = int(record.EventDataFields["ProcessID"], 10)
            node_id = self.map_pid(pid)
        elif record.EventID == 4689: # exit
            pid = int(record.EventDataFields["ProcessId"], 16)
            node_id = self.map_pid(pid)
            self.remove_pid(pid)
        else:
            try:
                pid = int(record.EventDataFields["ProcessId"], 16)
                node_id = self.map_pid(pid)
            except Exception:
                # some events (e.g., log cleared) have no process
                # None is guaranteed to never be in node_ids
                node_id = None

        if self.labeldir is not None:
            label = (node_id in self.attack_node_ids or
                     record.EventRecordID in self.attack_record_ids)
        else:
            label = None

        if label:
            assert node_id is not None
            self.attack_node_ids.add(node_id)
            self.attack_event_uuids.add(edge_uuid)

        self.writer.add_event(time, event, proc_name, str(node_id), label)

    def set_current_logfile(self, path: str) -> None:
        """Because EventRecordID is only unique within one log file, this
        needs to ensure the correct labels are being associated with the
        current event stream."""
        _, basename = path.rsplit("/", maxsplit=1)
        basename, _ = path.rsplit(".", maxsplit=1)
        _, _, host, atk, *_ = basename.split("-")
        if self.labeldir is not None:
            try:
                with open(f"{self.labeldir}/{host}_{atk}", "r") as f:
                    self.attack_record_ids = set(int(line) for line in f)
                    log.info("Using ground truth of %s_%s (%s events)",
                             host, atk, f"{len(self.attack_record_ids):,}")
            except FileNotFoundError:
                log.warning("No ground truth for %s_%s; assuming all benign...",
                            host, atk)
                self.attack_record_ids = set()

        # used to generate event UUIDs
        typ = path.rsplit("-", maxsplit=1)[1].split(".", maxsplit=1)[0]
        if typ.startswith("day"):
            typ = "b" + typ[-1]
        else:
            # s or m = attacks, reverted a snapshot
            self.reset_state()
        # typ is now either b[1-5] | s[1-4] | m[1-6]
        self.event_uuid_prefix = (b"\0" * 4) + typ.encode() + self.host.encode()
        assert len(self.event_uuid_prefix) == 8

def make_edge_uuid_pname_map(g: ig.Graph) -> Dict[str, str]:
    """Get the associated process with each edge in `g`.

    Rather than having to handle each edge based on its type to find the
    process node, just examine the source first, then the destination second.
    There are rarely some nodes in ATLASv2 that go from file to file (probably
    a parsing issue?); just return an unknown name for those."""
    ret = {}
    for e in g.es:
        src = g.vs[e.source]
        dst = g.vs[e.target]
        if src["type"] == "process":
            ret[e["uuid"]] = src["name"]
        elif dst["type"] == "process":
            ret[e["uuid"]] = dst["name"]
        else:
            log.warning("edge between two non-processes:")
            log.warning("[%s] -> [%s]", src["name"], dst["name"])
            ret[e["uuid"]] = parse.graph.unknown_proc
    return ret

def write_events(base_path: str, host: str) -> None:
    set_name = host
    events_path = f"{config.atlasv2.events}/{set_name}.csv.xz"

    if util.skip_overwrite(events_path):
        return

    graph_path = f"{config.atlasv2.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    log.info("mapping edge UUIDs to process names...")
    d = make_edge_uuid_pname_map(g)
    del g # try to free up space

    log.info("reading events of %s...", set_name)
    with lzma.open(events_path, "wt") as f:
        with ParseState(set_name, d, f) as state:
            json_stream.parse_json_set(base_path, host, state)
            log.info("finished %s; total events: %d.", set_name, state.writer.n)

def write_labeled_events(base_path: str, host: str, labeldir: str) -> None:
    set_name = host
    events_path = f"{config.atlasv2.events}/{set_name}_labeled.csv.xz"

    if util.skip_overwrite(events_path):
        return

    graph_path = f"{config.atlasv2.graphs}/{set_name}_prov_graph.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    log.info("mapping edge UUIDs to process names...")
    d = make_edge_uuid_pname_map(g)
    del g # try to free up space

    log.info("reading labeled events of %s...", set_name)
    with lzma.open(events_path, "wt") as f:
        with ParseState(set_name, d, f, labeldir=labeldir) as state:
            json_stream.parse_json_set(base_path, host, state)
            attack_event_uuids = state.attack_event_uuids
            log.info("finished %s; total events: %d.", set_name, state.writer.n)

    # write attack event UUIDs, if needed
    label_path = f"{config.atlasv2.labels}/{set_name}_attack_event_uuids.xz"
    with lzma.open(label_path, "wt") as f:
        for uuid in attack_event_uuids:
            f.write(str(uuid))
            f.write("\n")

def main(argv) -> None:
    base = "<PATH_TO_DATASET>/atlasv2"
    labeldir = "<PATH_TO_REAPR>/atlasv2"
    for host in ["h1", "h2"]:
        #write_events(base, host)
        write_labeled_events(base, host, labeldir)
