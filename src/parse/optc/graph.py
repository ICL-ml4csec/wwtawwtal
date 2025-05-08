"""Parse OpTC ecar logs into provenance graphs.

Note that for consistency with other datasets (such as E3), we don't handle
some of the Windows-specific behaviors that are in OpTC, such as processes
opening handles to other processes."""

import concurrent.futures
import enum
import gc
import logging
from typing import Dict

import dateutil.parser

import config
from parse import graph
import parse.optc.json_stream
import util

log = logging.getLogger(__name__)

class ParseState(parse.optc.json_stream.AbstractParseState):
    def __init__(self, host_nbr: int):
        super().__init__(host_nbr)
        self.g = graph.GraphBuilder()
        self.node_ids = {} # actorID/objectID -> int
        self.next_id = 0
        self.doc2vec_names = set()

        self.object_handlers = {
            "FILE": self.parse_file,
            "FLOW": self.parse_flow,
            "MODULE": self.parse_module,
            "PROCESS": self.parse_process,
            "REGISTRY": self.parse_registry,
            "THREAD": self.parse_thread,
        }
        self.ignore_objects = {
            "SERVICE", #
            "TASK",    # these might actually be better to track...
            "HOST", # indicates when a system (re)starts
            "SHELL", # this seems to literally indicate the attack
            "USER_SESSION",
        }

    def map_id(self, uuid: str) -> int:
        """Map the given UUID (generally actorID for processes or objectID for
        files/sockets) to a node id, and return the mapped id."""
        if uuid not in self.node_ids:
            self.node_ids[uuid] = self.next_id
            self.next_id += 1
        return self.node_ids[uuid]

    def add_new_node(self, node: int, time: int, **kwargs):
        """Add `node` to `self.g` only if it does not already exist."""
        self.g.add_node(node, time=time, overwrite=False, **kwargs)

    def add_node(self, node: int, time: int, **kwargs):
        """Add `node` to `self.g`, overwriting any previous node."""
        self.g.add_node(node, time=time, overwrite=True, **kwargs)

    def add_edge(self, src: int, dst: int, **kwargs):
        """Add an edge from `src` to `dst` in `self.g`."""
        self.g.add_edge(src, dst, **kwargs)

    def get_time(self, record: Dict) -> int:
        """Return a nanosecnod timestamp from record["timestamp"]."""
        dt = dateutil.parser.isoparse(record["timestamp"])
        return int(dt.timestamp() * 1e9)

    def parse_record(self, record: Dict) -> None:
        # delegate to object handlers
        if record["object"] in self.object_handlers:
            self.object_handlers[record["object"]](record)
        elif record["object"] in self.ignore_objects:
            return
        else:
            raise ValueError(f"unknown object type: {record['object']}")

    # break parsers by object type
    def parse_flow(self, record: Dict) -> None:
        # Flow objectIDs appear to be based on the source-dest addreses.
        # Interestingly, two different hosts with the same IP addresses (???)
        # can end up having the same flow objectID.
        # Unlike the E3 parser, this will only create one socket node per IP
        # address, but we have found this to be ideal behavior.
        #
        # I chose to map "start" flows to "connect" events, which seems
        # corroborated by the lack of a "size" field in start events.
        props = record["properties"]
        if props["direction"] == "inbound":
            is_recv = True
        elif props["direction"] == "outbound":
            is_recv = False
        else:
            raise ValueError(f"invalid direction: {props['direction']}")

        if record["action"] == "START":
            typ = "connect"
        elif record["action"] == "MESSAGE":
            typ = "recv" if is_recv else "send"
        elif record["action"] == "OPEN":
            # don't actually know what these are... they have source IPs but
            # not destinations, so skip for now
            return
        else:
            raise ValueError(f"unhandled action: {record['action']}")

        time = self.get_time(record)
        uuid = record["id"]
        proc_uuid = record["actorID"]
        sock_uuid = record["objectID"]
        proc_id = self.map_id(proc_uuid)
        sock_id = self.map_id(sock_uuid)
        try:
            proc_name = props["image_path"]
        except KeyError:
            proc_name = parse.graph.unknown_proc
        if is_recv:
            sock_name = f"sock={props['dest_ip']}:{props['dest_port']}"
        else:
            sock_name = f"sock={props['src_ip']}:{props['src_port']}"

        self.add_new_node(proc_id, name=proc_name, uuid=proc_uuid,
                          type="process", time=time)
        self.add_new_node(sock_id, name=sock_name, uuid=sock_uuid,
                          type="socket", time=time)
        if is_recv:
            self.add_edge(sock_id, proc_id, type=typ, time=time, uuid=uuid)
        else:
            self.add_edge(proc_id, sock_id, type=typ, time=time, uuid=uuid)

        self.doc2vec_names.add(proc_name)
        self.doc2vec_names.add(sock_name)

    def parse_process(self, record: Dict):
        props = record["properties"]
        if record["action"] in {"OPEN", "TERMINATE"}:
            # ignore these
            return
        elif record["action"] == "CREATE":
            # CreateProcess; alias to "clone" to match existing graphs
            typ = "clone"
        else:
            raise ValueError(f"unhandled action: {record['action']}")

        time = self.get_time(record)
        uuid = record["id"]
        parent_uuid = record["actorID"]
        child_uuid = record["objectID"]
        parent_id = self.map_id(parent_uuid)
        child_id = self.map_id(child_uuid)
        try:
            parent_name = props["parent_image_path"]
        except KeyError:
            parent_name = parse.graph.unknown_proc
        try:
            child_name = props["image_path"]
        except KeyError:
            parent_name = parse.graph.unknown_proc

        self.add_new_node(parent_id, name=parent_name, uuid=parent_uuid,
                          type="process", time=time)
        self.add_node(child_id, name=child_name, uuid=child_uuid,
                      type="process", time=time)
        self.add_edge(parent_id, child_id, type=typ, time=time, uuid=uuid)

        self.doc2vec_names.add(parent_name)
        self.doc2vec_names.add(child_name)

    def parse_file(self, record: Dict):
        props = record["properties"]
        if record["action"] in {"CREATE", "WRITE", "MODIFY"}:
            # not sure what modify is, just treat it same as write
            # also just group CREATE with WRITE
            typ = "write"
            is_write = True
        elif record["action"] == "DELETE":
            typ = "delete"
            is_write = True
        elif record["action"] == "READ":
            typ = "read"
            is_write = False
        elif record["action"] == "RENAME":
            return # ignore
        else:
            raise ValueError(f"unhandled action: {record['action']}")

        time = self.get_time(record)
        uuid = record["id"]
        proc_uuid = record["actorID"]
        file_uuid = record["objectID"]
        proc_id = self.map_id(proc_uuid)
        file_id = self.map_id(file_uuid)
        try:
            proc_name = props["image_path"]
        except KeyError:
            proc_name = parse.graph.unknown_proc
        try:
            file_name = props["file_path"]
        except KeyError:
            file_name = parse.graph.unknown_file

        self.add_new_node(proc_id, name=proc_name, uuid=proc_uuid,
                          type="process", time=time)
        self.add_new_node(file_id, name=file_name, uuid=file_uuid,
                          type="file", time=time)
        if is_write:
            self.add_edge(proc_id, file_id, type=typ, time=time, uuid=uuid)
        else:
            self.add_edge(file_id, proc_id, type=typ, time=time, uuid=uuid)

        self.doc2vec_names.add(proc_name)
        self.doc2vec_names.add(file_name)

    def parse_module(self, record: Dict) -> None:
        props = record["properties"]
        if record["action"] == "LOAD":
            typ = "loadlibrary"
        else:
            raise ValueError(f"unhandled action: {record['action']}")

        time = self.get_time(record)
        uuid = record["id"]
        proc_uuid = record["actorID"]
        lib_uuid = record["objectID"]
        proc_id = self.map_id(proc_uuid)
        lib_id = self.map_id(lib_uuid)
        try:
            proc_name = props["image_path"]
        except KeyError:
            proc_name = parse.graph.unknown_proc
        try:
            lib_name = props["module_path"]
        except KeyError:
            proc_name = parse.graph.unknown_file

        self.add_new_node(proc_id, name=proc_name, uuid=proc_uuid,
                          type="process", time=time)
        self.add_new_node(lib_id, name=lib_name, uuid=lib_uuid,
                          type="file", time=time)
        self.add_edge(lib_id, proc_id, type="loadlibrary", time=time, uuid=uuid)

        self.doc2vec_names.add(proc_name)
        self.doc2vec_names.add(lib_name)

    def parse_registry(self, record: Dict) -> None:
        props = record["properties"]
        if record["action"] in {"ADD", "EDIT"}:
            typ = "write"
            is_write = True
        elif record["action"] == "REMOVE":
            typ = "delete"
            is_write = True
        else:
            raise ValueError(f"unhandled action: {record['action']}")

        time = self.get_time(record)
        uuid = record["id"]
        proc_uuid = record["actorID"]
        regkey_uuid = record["objectID"]
        proc_id = self.map_id(proc_uuid)
        regkey_id = self.map_id(regkey_uuid)
        try:
            proc_name = props["image_path"]
        except KeyError:
            proc_name = parse.graph.unknown_proc
        try:
            regkey_name = props["key"]
        except KeyError:
            regkey_name = parse.graph.unknown_file

        self.add_new_node(proc_id, name=proc_name, uuid=proc_uuid,
                          type="process", time=time)
        self.add_new_node(regkey_id, name=regkey_name, uuid=regkey_uuid,
                          type="regkey", time=time)
        if is_write:
            self.add_edge(proc_id, regkey_id, type=typ, time=time, uuid=uuid)
        else:
            self.add_edge(regkey_id, proc_id, type=typ, time=time, uuid=uuid)

        self.doc2vec_names.add(proc_name)
        self.doc2vec_names.add(regkey_name)

    def parse_thread(self, record: Dict) -> None:
        props = record["properties"]
        if record["action"] == "TERMINATE":
            # ignore
            return
        elif record["action"] in {"CREATE", "REMOTE_CREATE"}:
            # CreateProcess; alias to "clone" to match existing graphs
            typ = "clone"
        else:
            raise ValueError(f"unhandled action: {record['action']}")

        time = self.get_time(record)
        uuid = record["id"]
        parent_uuid = record["actorID"]
        child_uuid = record["objectID"]
        parent_id = self.map_id(parent_uuid)
        child_id = self.map_id(child_uuid)
        try:
            name = props["image_path"]
        except KeyError:
            name = parse.graph.unknown_proc

        self.add_new_node(parent_id, name=name, uuid=parent_uuid,
                          type="thread", time=time)
        self.add_node(child_id, name=name, uuid=child_uuid,
                      type="thread", time=time)
        self.add_edge(parent_id, child_id, type=typ, time=time, uuid=uuid)

        self.doc2vec_names.add(name)

    def is_empty(self) -> bool:
        """Return whether this graph contains any nodes or not."""
        return self.next_id == 0

def _write_graph(state: ParseState, host: int) -> None:
    if not state.is_empty():
        names_path = f"{config.optc.names}/host{host}_prov_graph.txt"
        log.info("writing all names to %s", names_path)
        with open(names_path, "wt") as f:
            for name in state.doc2vec_names:
                f.write(name)
                f.write("\n")

        log.info("building provenance graph")
        g = state.g.build()
        log.info("  |V| = %d, |E| = %d", g.vcount(), g.ecount())

        del state

        graph_path = f"{config.optc.graphs}/host{host}_prov_graph.pkl.xz"
        log.info("writing %s...", graph_path)
        util.pickle_write(g, graph_path)
        log.info("done")

def write_graph(host_nbr: int) -> None:
    """Construct the provenance graph for host `host_nbr` and write it to
    intermediates."""
    state = ParseState(host_nbr)
    log.info("parsing graph for host %i", host_nbr)
    parse.optc.json_stream.parse_host(state)
    log.info("finished.")

    _write_graph(state, host_nbr)


def write_batch(start: int) -> None:
    """Batch the construction of 25 graphs starting from `start`. `start`
    should be a (multiple of 25) + 1, e.g., 1, 26, 51, etc."""
    assert start % 25 == 1
    end = start + 24
    states = {h: ParseState(h) for h in range(start, end + 1)}
    logdir = f"AIA-{start}-{end}"

    log.info("parsing graphs in range [%i, %i]", start, end)
    parse.optc.json_stream.parse_batch(states, logdir)
    log.info("finished.")

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        for h in range(start, end + 1):
            state = states[h]
            futures.append(pool.submit(_write_graph, state, h))
            del states[h]

        while futures:
            _, futures = concurrent.futures.wait(futures)

def main(argv) -> None:
    # 201, 501, and 51 are the specific hosts
    #write_graph(51)
    #write_graph(201)
    #write_graph(501)
    # pick a random benign host: 142
    #write_graph(321)
    #hosts = [201, 402, 660, 501, 811, 974, 5, 51]
    #for host in hosts:
    #    write_graph(host)

    # parse everything
    #for start in range(1, 975, 25):
    for start in range(901, 975, 25):
        write_batch(start)
        gc.collect()
