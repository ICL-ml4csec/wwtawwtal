"""Parsing raw SystemTap data (found in
/shared/projects/2016-streamspot-dataset/raw-stap-outputs/stap_outputs/*)."""

from dataclasses import dataclass
import enum
import glob
import lzma
import logging
import os
import re
from typing import Dict, Iterator, List, Optional, TextIO, Tuple, Union

import igraph as ig
import tqdm

import config
import parse.graph
import util

log = logging.getLogger(__name__)

@dataclass
class Event:
    """A SystemTap event from the StreamSpot dataset.

    Note that the timestamp unit is unlabelled. Based on the final timestamp
    ending just before 2.4e10, the unit is probably nanoseconds, corresponding
    to ~24 seconds per trial. The alternative, microseconds, would correspond
    to ~6.7 hours per trial; for 100 trials just per dataset, this seems
    unlikely."""
    ret_val: int
    ret_time: int
    call_time: int
    process_name: str
    pid: int
    tid: int
    syscall: str
    args: str

# Parsing with parsec is extremely slow, so we're back to ad-hoc parsing
# Fortunately, the systap data is very well-structured so it's not so bad
def parse_line(line: str) -> Optional[Event]:
    try:
        fields = line.split(", ", maxsplit=7)
        fields[0] = int(fields[0].split(" ", maxsplit=1)[0], 0) # ret_val
        fields[1] = int(fields[1], 0) # ret_time
        fields[2] = int(fields[2], 0) # call_time
        # process_name
        fields[4] = int(fields[4], 0) # pid
        fields[5] = int(fields[5], 0) # tid
        # syscall
        # args

        return Event(*fields)
    except ValueError:
        return None

def parse_file(path: str) -> List[Event]:
    with open(path, "r") as f:
        next(f) # ignore header
        return list(filter(lambda x: x is not None,
                           (parse_line(line) for line in f)))

class TimeMode(enum.Enum):
    VARIABLE = enum.auto() # variable bins => constant time between trials
    FIXED = enum.auto() # fixed bins => variable time between trials

class ParseState:
    """Parsing state, e.g., timestamps and numeric IDs between trials of a
    dataset. Note that some state should be global to all parsed datasets
    (e.g., paths used to train Doc2Vec), while others are per-dataset (e.g.,
    numeric IDs used in the provenance graph)."""
    def __init__(self, time_mode=TimeMode.VARIABLE):
        # global state
        self.doc2vec_names = set()
        if not isinstance(time_mode, TimeMode):
            raise ValueError("invalid time_mode")
        self.time_mode = time_mode

    def reset_id_mappings(self) -> None:
        """Only reset ID mappings; all other state remains intact."""
        self.node_ids = {} # node -> id (both filename -> id, pid -> id)
        self.proc_fds = {} # pid -> file descriptor -> {id, file name}

    def reset_dataset_state(self, path: Optional[str] = None) -> None:
        """(Re)set the per-dataset state, e.g., ID mappings."""
        self.g = parse.graph.GraphBuilder()
        self.next_id = 0

        if path is not None:
            _, name = path.rsplit("/", maxsplit=1)
            self.is_attack = name == "attack"
        else:
            self.is_attack = False
        self.attack_nodes = set() # set of malicious node IDs for attack
        self.attack_edges = set() # malicious edge UUIDs

        # each trial starts at an arbitrary time; add offset to order them
        # before a trial, this marks the time when the trial starts
        # within a trial, this shifts the times to start at the trial's start
        #
        # VARIABLE MODE
        # =============
        # e.g., suppose first trial spans [100, 200], second spans [80, 180].
        # before first trial: offset = 0.
        # within first trial: offset = -100
        #   now `time + time_offset` shifts the trial to
        #   [0, 100]
        #
        # at end of first trial: set offset = 100 + time_between_trials
        #
        # before second trial: offset = 100 + time_between_trials
        # within second trial: offset = 20 + time_between_trials
        #   now `time + time_offset` shifts the trial to
        #   [100 + time_between_trials, 200 + time_between_trials]
        #
        # at end of second trial: set offset = 200 + 2 * time_between_trials
        # etc.
        #
        # therefore:
        #   when a trial starts:
        #     set offset -= start_of_trial
        #   when a trial ends:
        #     set offset = shifted_final_time + time_between_trials
        #
        # FIXED MODE
        # ==========
        # This is much easier. The known maximum trial length is around 95
        # seconds; use 100 seconds for a round number. Simply set offset to
        # (100 * i_trial) - start_of_trial
        self.time_offset = 0

        # 1 us = 1000 ns between datasets is around the spacing of events
        self.time_between_trials = 1000

        # record timestamps between trials
        self.trial_boundaries = []

    def first_event_set_time_offset(self, first: Event) -> None:
        """Set time_offset so all times within the dataset are correct."""
        if self.time_mode == TimeMode.VARIABLE:
            self.time_offset -= first.call_time
        elif self.time_mode == TimeMode.FIXED:
            self.time_offset = (100e9 * len(self.trial_boundaries)
                                - first.call_time)
        else:
            raise ValueError("invalid TimeMode")

    def last_event_set_time_offset(self, last: Event) -> None:
        """Set time_offset so the next dataset will be correct."""
        if self.time_mode == TimeMode.VARIABLE:
            self.trial_boundaries.append(
                    self.time(last) + self.time_between_trials / 2)
            self.time_offset = self.time(last) + self.time_between_trials
        elif self.time_mode == TimeMode.FIXED:
            # put the boundary 99 seconds after start of current
            # known to be safe because longest trial is 95 seconds, and bins
            # are 100 seconds
            self.trial_boundaries.append(len(self.trial_boundaries) * 100e9
                                         + 99e9)
        else:
            raise ValueError("invalid TimeMode")

    def time(self, event: Event) -> int:
        """Return the corrected timestamp for `event`."""
        return self.time_offset + event.call_time

    def map_id(self, obj: Union[int, str]) -> int:
        """Map obj to a unique id, and return the mapped id."""
        if obj not in self.node_ids:
            self.node_ids[obj] = self.next_id
            self.next_id += 1
        return self.node_ids[obj]

    def add_new_node(self, node: int, type: str, name: str, time: int,
                     pid: Optional[int] = None):
        """Add `node` to `self.g` only if it does not already exist.
        Note that this uses the node's integer ID as its UUID since it's
        guaranteed to be unique within a dataset anyways."""
        uuid = f"node={node}" if pid is None else f"pid={pid}"
        self.g.add_node(node, type, name, time, uuid=uuid, overwrite=False)

    def add_node(self, node: int, type: str, name: str, time: int,
                 pid: Optional[int] = None):
        uuid = f"node={node}" if pid is None else f"pid={pid}"
        self.g.add_node(node, type, name, time, uuid=uuid, overwrite=True)

    def add_edge(self, src: int, dst: int, type: str, time: int, uuid: str):
        """Add an edge, and also propagate attack labels."""
        self.g.add_edge(src, dst, type, time, uuid)
        if self.is_attack and src in self.attack_nodes:
            self.attack_nodes.add(dst)

    def mark_attack_event(self, line: int, event: Event):
        """Mark `event` as being an attack event if its TID is an attack node."""
        try:
            proc_id = self.node_ids[event.pid]
        except KeyError:
            return
        if self.is_attack and proc_id in self.attack_nodes:
            self.attack_edges.add(self.edge_uuid(line, event))

    def edge_uuid(self, line: int, event: Event) -> str:
        """Create an edge UUID for an event. This is just the current trial
        number and file line number, separated by a hyphen."""
        trial = len(self.trial_boundaries)
        return f"{trial}-{line}"

    def parse_clone(self, line: int, event: Event) -> None:
        if event.ret_val < 0:
            return
        # skip threads
        # they use shared memory and shouldn't be modeled as separate nodes
        if "CLONE_THREAD" in event.args:
            return

        src_id = self.map_id(event.pid)
        dst_id = self.map_id(event.ret_val)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        node_attrs = {
            "type": "process",
            "time": time,
        }

        self.add_new_node(src_id, name=event.process_name, pid=event.pid,
                          **node_attrs)
        # cloned uses same name as parent
        name = self.g.node_attrs[src_id].name
        self.add_new_node(dst_id, name=name, pid=event.ret_val, **node_attrs)

        self.doc2vec_names.add(name)

        self.add_edge(src_id, dst_id, type="clone", time=time, uuid=edge_uuid)

    def parse_execve(self, line: int, event: Event) -> None:
        """Parse an execve event to learn the correct process name. This
        overwrites a node's ID, although technically, it is possible that exec
        could represent two distinct processes."""
        if event.ret_val < 0:
            return
        n = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)
        # the first arg is either a path to the file to execute, or a pointer
        # every time it is a pointer, the resulting process is sh
        # all shells are sh; there is no bash, /bin/sh, etc.
        match = re.match('"([^"]*)"', event.args)
        if match:
            name = match.group(1)
        else:
            name = "sh"
        # intentionally overwrite any existing node metadata
        # we want this process's start time to be when it execs
        self.add_node(n, name=name, type="process", pid=event.pid, time=time)

        # easy way to find attack shell behavior
        if self.is_attack and "ps x" in event.args:
            self.attack_nodes.add(n)
            self.attack_edges.add(edge_uuid)

        self.doc2vec_names.add(name)

    def parse_connect(self, line: int, event: Event) -> None:
        """Parse a connect event; for attack, also look for point of entry."""
        # -115 (EINPROGRESS) is not an error
        if event.ret_val < 0 and event.ret_val != -115:
            return

        proc_id = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        try:
            fd, family, addr, maybe_port, *rest = event.args.split(", ")
        except ValueError:
            # unknown sockaddr, give up
            return

        family = family[1:] # remove { of struct sockaddr
        if family == "AF_UNIX":
            addr = addr[:-1] # remove } of struct sockaddr
        elif family == "AF_INET":
            port = maybe_port[:-1] # remove } of struct sockaddr
            addr = f"{addr}:{port}"
        elif family.startswith("unknown"):
            return # skip
        else:
            log.warning("unknown family %s; skipping...", family)
            return
            #raise ValueError("unknown family")

        # prepend sock= to addr to normalize format
        addr = f"sock={addr}"
        sock_id = self.map_id(addr)
        if proc_id not in self.proc_fds:
            self.proc_fds[proc_id] = {}
        self.proc_fds[proc_id][fd] = (sock_id, addr)

        self.add_new_node(sock_id, name=addr, type="socket", time=time)
        self.doc2vec_names.add(addr)

        if self.is_attack and addr.startswith("192.168.0.100"):
            self.attack_nodes.add(proc_id)
            self.attack_nodes.add(sock_id)
            # maybe the act of connecting itself shouldn't be malicious
            self.attack_edges.add(edge_uuid)

        #self.add_edge(sock_id, proc_id, type="connect", time=time, uuid=edge_uuid)

    def parse_open(self, line: int, event: Event) -> None:
        if event.ret_val < 0:
            return

        proc_id = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        path, flags = event.args.split(", ", maxsplit=1)
        fd = event.ret_val
        file_id = self.map_id(path)
        if proc_id not in self.proc_fds:
            self.proc_fds[proc_id] = {}
        self.proc_fds[proc_id][fd] = (file_id, path)

        self.add_new_node(file_id, name=path, type="file", time=time)
        self.doc2vec_names.add(path)

    def parse_write(self, line: int, event: Event) -> None:
        if event.ret_val < 0:
            return

        proc_id = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        fd, _ = event.args.split(", ", maxsplit=1)

        try:
            file_id, _ = self.proc_fds[proc_id][fd]
        except KeyError:
            # we don't know this file, give up
            return

        self.add_edge(proc_id, file_id, type="write", time=time, uuid=edge_uuid)

    def parse_read(self, line: int, event: Event) -> None:
        if event.ret_val < 0:
            return

        proc_id = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        fd, _ = event.args.split(", ", maxsplit=1)

        try:
            file_id, _ = self.proc_fds[proc_id][fd]
        except KeyError:
            # we don't know this file, give up
            return

        self.add_edge(file_id, proc_id, type="read", time=time, uuid=edge_uuid)

    def parse_send(self, line: int, event: Event) -> None:
        """This is almost unnecessary, but UNIX sockets may propagate"""
        if event.ret_val < 0:
            return

        proc_id = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        fd, _ = event.args.split(", ", maxsplit=1)

        try:
            file_id, _ = self.proc_fds[proc_id][fd]
        except KeyError:
            # we don't know this file, give up
            return

        self.add_edge(proc_id, file_id, type="send", time=time, uuid=edge_uuid)

    def parse_recv(self, line: int, event: Event) -> None:
        if event.ret_val < 0:
            return

        proc_id = self.map_id(event.pid)
        time = self.time(event)
        edge_uuid = self.edge_uuid(line, event)

        fd, _ = event.args.split(", ", maxsplit=1)

        try:
            file_id, _ = self.proc_fds[proc_id][fd]
        except KeyError:
            # we don't know this file, give up
            return

        self.add_edge(file_id, proc_id, type="recv", time=time, uuid=edge_uuid)

    # to avoid redundant work, just do processes only for now
    handlers = {
        "clone": parse_clone,
        "execve": parse_execve,
        "connect": parse_connect,
        "open": parse_open,
        "write": parse_write,
        "writev": parse_write,
        "read": parse_read,
        "readv": parse_read,
        "send": parse_send,
        "sendto": parse_send,
        "sendmsg": parse_send,
        "recv": parse_recv,
        "recvfrom": parse_recv,
        "recvmsg": parse_recv,
    }

    def parse_event(self, line: int, event: Event) -> None:
        """Add the data of `event` to the provenance graph."""
        self.mark_attack_event(line, event)
        if event.syscall in ParseState.handlers:
            try:
                ParseState.handlers[event.syscall](self, line, event)
            except:
                log.error("Error at event: %s", event)
                raise

class ProvenanceGraph:
    """The provenance graph with extra metadata. This is essentially just a
    wrapper around ig.Graph with extra data, like the true timestamps
    between trial boundaries."""
    def __init__(self, state: ParseState):
        self.g = state.g.build()
        self.trial_boundaries = state.trial_boundaries

def _event_generator(f: TextIO) -> Iterator[Tuple[int, Event]]:
    # start at 1 because header was removed
    for i, line in enumerate(f, start=1):
        event = parse_line(line)
        if event is not None:
            yield i, event

def _parse_file_graph(path: str, state: ParseState) -> None:
    with open(path, "r") as f:
        next(f)
        gen = _event_generator(f)
        line, event = next(gen)
        if event is None:
            return

        state.first_event_set_time_offset(event)
        state.parse_event(line, event)

        for line, event in gen:
            state.parse_event(line, event)

        state.last_event_set_time_offset(event)

def extract_trial_number(path: str) -> int:
    _, n = path.rsplit("/", maxsplit=1)
    try:
        return int(n)
    except ValueError:
        return 0 # this file doesn't matter anyways

def parse_set_graph(path: str, state: ParseState) -> ProvenanceGraph:
    state.reset_dataset_state(path)

    files = sorted(glob.iglob(f"{path}/*"), key=extract_trial_number)
    for name in tqdm.tqdm(files,
                          desc=os.path.basename(path), leave=None):
        # there's some other temp files mixed in, skip them
        base = os.path.basename(name)
        if base.isnumeric():
            _parse_file_graph(name, state)
    # trim off the last boundary
    state.trial_boundaries.pop()
    return ProvenanceGraph(state)

def dump_pickle_dataset(path: str, state: ParseState) -> None:
    basename = os.path.basename(path.rstrip("/"))
    pickle_path = f"{config.streamspot.graphs}/{basename}_prov_graph.pkl.xz"

    # parsing a graph is slow, so open the file first (which will cause an
    # error if it cannot be opened), and also check if it exists to avoid
    # overwriting
    if util.skip_overwrite(pickle_path):
        return

    graph = parse_set_graph(path, state)
    util.pickle_write(graph, pickle_path)

def pickle_datasets(basedir: str) -> None:
    # check this first
    doc2vec_path = f"{config.streamspot.intermediates}/doc2vec_names.txt"
    if util.skip_overwrite(doc2vec_path):
        return

    state = ParseState()
    for d in tqdm.tqdm(glob.glob(f"{basedir}/*")):
        base = os.path.basename(d)
        if base == "continuousParser":
            continue
        dump_pickle_dataset(d, state)

    with open(doc2vec_path, "w") as f:
        for path in state.doc2vec_names:
            f.write(f"{path}\n")

def label_attack(basedir: str) -> None:
    state = ParseState()
    path = f"{basedir}/attack"
    parse_set_graph(path, state)

    def uuid_key(uuid: str) -> Tuple[int, int]:
        trial, line = uuid.split("-", maxsplit=1)
        return int(trial), int(line)

    out_path = f"{config.streamspot.labels}/attack_edges.xz"
    with lzma.open(out_path, "wt") as f:
        for uuid in sorted(state.attack_edges, key=uuid_key):
            f.write(uuid)
            f.write("\n")

    label_path = f"{config.streamspot.labels}/attack_nodes.xz"
    with lzma.open(label_path, "wt") as f:
        for nid in sorted(state.attack_nodes):
            f.write(str(nid))
            f.write("\n")

def write_gnn_ablation_graph(basedir: str) -> None:
    benigns = ("cnn", "download", "game", "gmail", "youtube")
    state = ParseState()
    state.reset_dataset_state()

    for trial in tqdm.tqdm(range(0, 100)):
        for benign in benigns:
            state.reset_id_mappings()
            path = f"{basedir}/{benign}/{trial}"
            _parse_file_graph(path, state)

    # add one attack file
    path = f"{basedir}/attack"
    state.reset_id_mappings()
    # turn on attack labeling
    state.is_attack = True

    _parse_file_graph(f"{path}/0", state)

    # trim off the last boundary
    state.trial_boundaries.pop()

    # write outputs
    graph_path = f"{config.streamspot.graphs}/gnn_ablation_prov_graph.pkl.xz"
    g = ProvenanceGraph(state)
    util.pickle_write(g, graph_path)

    def uuid_key(uuid: str) -> Tuple[int, int]:
        trial, line = uuid.split("-", maxsplit=1)
        return int(trial), int(line)

    out_path = f"{config.streamspot.labels}/gnn_ablation_edges.xz"
    with lzma.open(out_path, "wt") as f:
        for uuid in sorted(state.attack_edges, key=uuid_key):
            f.write(uuid)
            f.write("\n")

    out_path = f"{config.streamspot.labels}/gnn_ablation_nodes.xz"
    with lzma.open(out_path, "wt") as f:
        for nid in sorted(state.attack_nodes):
            f.write(str(nid))
            f.write("\n")

def write_all_graph(basedir: str) -> None:
    # striped benigns -> all attacks
    benigns = ("cnn", "download", "game", "gmail", "youtube")
    state = ParseState(time_mode=TimeMode.FIXED)
    state.reset_dataset_state()

    for trial in tqdm.tqdm(range(0, 100)):
        for benign in benigns:
            state.reset_id_mappings()
            path = f"{basedir}/{benign}/{trial}"
            _parse_file_graph(path, state)

    # for convenience: record the end of benign timestamp
    benign_end_time = state.trial_boundaries[-1]

    # add attacks
    # turn on attack labeling
    state.is_attack = True
    for trial in tqdm.tqdm(range(0, 100)):
        state.reset_id_mappings()
        path = f"{basedir}/attack/{trial}"
        _parse_file_graph(path, state)

    # trim off the last boundary
    state.trial_boundaries.pop()

    # write outputs
    graph_path = f"{config.streamspot.graphs}/all_prov_graph.pkl.xz"
    g = ProvenanceGraph(state)
    g.benign_end_time = benign_end_time
    util.pickle_write(g, graph_path)

    def uuid_key(uuid: str) -> Tuple[int, int]:
        trial, line = uuid.split("-", maxsplit=1)
        return int(trial), int(line)

    out_path = f"{config.streamspot.labels}/all_edges.xz"
    with lzma.open(out_path, "wt") as f:
        for uuid in sorted(state.attack_edges, key=uuid_key):
            f.write(uuid)
            f.write("\n")

    out_path = f"{config.streamspot.labels}/all_nodes.xz"
    with lzma.open(out_path, "wt") as f:
        for nid in sorted(state.attack_nodes):
            f.write(str(nid))
            f.write("\n")

def pickle_attack_0() -> None:
    state = ParseState()
    dump_pickle_dataset("streamspot-attack0", state)

def main(argv: List[str]) -> None:
    basedir = "<PATH_TO_STREAMSPOT_STAP>"
    #pickle_datasets(basedir)
    #label_attack(basedir)
    #write_gnn_ablation_graph(basedir)
    write_all_graph(basedir)
