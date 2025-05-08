import csv
import enum
import logging
import lzma
from typing import NamedTuple, Optional, Set, Tuple
import uuid

import config
from parse import graph
from parse.atlasv2 import json_stream
import util

log = logging.getLogger(__name__)

def read_labels(label_path: str) -> dict:
    """Read the ATLASv2 labels from `label_path`, and return a dict mapping
    `pid-name` to "benign", "contaminated", or "attack".

    Note the format is `logfile, pid, process_name, label`, with the
    extraneous spaces. pid-name is not actually a unique identifier, but
    should hopefully never result in conflation."""
    out = {}
    fieldnames = ("logfile", "pid", "name", "type")
    with open(label_path, "r", newline="") as f:
        reader = csv.DictReader(f, fieldnames, skipinitialspace=True)
        for row in reader:
            logfile = row["logfile"]
            if logfile not in out:
                out[logfile] = {}
            key = f"{row['pid']}-{row['name']}"
            if key in out[logfile] and out[logfile][key] != row["type"]:
                log.warning("%s-%s-%s: conflict, marking as contaminated",
                            row["logfile"], row["pid"], row["name"])
                out[logfile][key] = "contaminated"
            else:
                out[logfile][key] = row["type"]
    return out

class AccessMask(enum.IntEnum):
    READ = 0x1
    WRITE = 0x2
    APPEND = 0x4
    READ_EA = 0x8
    WRITE_EA = 0x10
    EXECUTE = 0x20
    DELETE_CHILD = 0x40
    READ_ATTR = 0x80
    WRITE_ATTR = 0x100
    DELETE = 0x10000
    READ_CONTROL = 0x20000
    WRITE_DAC = 0x40000
    WRITE_OWNER = 0x80000
    SYNCHRONIZE = 0x100000
    ACCESS_SYS_SEC = 0x1000000

    # pretty sure library loads are EXECUTE
    PROV_READ = READ
    PROV_WRITE = WRITE | APPEND | DELETE_CHILD
    PROV_EXEC = EXECUTE

class NetDirection(str, enum.Enum):
    IN = "%%14592"
    OUT = "%%14593"

class ParseState(json_stream.AbstractParseState):
    def __init__(self, host: str, labels: dict = {}):
        super().__init__(host)
        # NB: Windows reuses PIDs as soon as they're available, so this must
        # track liveness. Since the logs are ordered in time, can use a simple
        # stateful approach
        self.pid_table = {}
        self.path_table = {}
        self.names = set()
        self.next_id = 0
        self.g = graph.GraphBuilder()

        # for labeling
        if not labels:
            log.warning("empty label dictionary!")
        self.all_labels = labels
        self.contaminated_uuids = set()
        self.attack_uuids = set()

        # event UUID prefix is [bsm][0-9]h[1-2]; node prefix is simply nh[1-2]
        self.node_uuid_prefix = (b"\0" * 5) + b"n" + host.encode()
        assert len(self.node_uuid_prefix) == 8

    def reset_state(self) -> None:
        """Reset mapping state. This is especially important for the attacks
        which were reverted from snapshots; the nodes will not correlate
        anymore.

        Calling this will result in multiple disjoint subgraphs."""
        # TODO better to smash the graphs together or separate?
        self.pid_table = {}
        self.path_table = {}

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

    def map_obj(self, path: str) -> int:
        """Map `obj` (a file path, registry key, or socket name) to a node ID.

        We track object paths since they are available, rather than tracking
        object handles per process. We can handle moves/renames by simply
        changing the path table."""
        if path not in self.path_table:
            # intentionally share the same node ID space with PIDs
            self.path_table[path] = self.next_id
            self.next_id += 1
        return self.path_table[path]

    @staticmethod
    def get_time(event: NamedTuple) -> int:
        return event.SystemTime

    def add_node(self, node: int, **kwargs):
        self.g.add_node(node, overwrite=True, **kwargs)

    def add_new_node(self, node: int, **kwargs):
        self.g.add_node(node, overwrite=False, **kwargs)

    def add_edge(self, src: int, dst: int, **kwargs):
        self.g.add_edge(src, dst, **kwargs)

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

    def make_node_uuid(self, node_id: int) -> uuid.UUID:
        """Like `make_edge_uuid`, but for nodes. Since each node already has a
        unique id, all we need to do is ensure the prefix avoids collisions
        with edges."""
        b = self.node_uuid_prefix + node_id.to_bytes(8, "big")
        return uuid.UUID(bytes=b)

    def rewrite_path(self, path: Optional[str]) -> Optional[str]:
        """Rewrite `path` to fix potential problems.

        Notably, some paths (like in Event 5156) use full device paths instead
        of Windows drives, which will probably be problematic. Assume that
        there is only one mounted drive, C:, corresponding to volume 1.
        """
        if path is None:
            return None
        # maybe this isn't necessary
        #vol = "\\device\\harddiskvolume1"
        #if path.startswith(vol):
        #    path = f"C:{path[len(vol):]}"
        return path

    def label_node(self, pid: int, name: str, uuid: str) -> None:
        """Possibly add `uuid` to the attack or contaminated sets, depending
        on whether or not `pid` and `name` correspond to a labeled process."""
        key = f"{pid}-{name.lower()}"
        label = self.labels.get(key)
        if label == "contaminated":
            self.contaminated_uuids.add(uuid)
        elif label == "attack":
            self.attack_uuids.add(uuid)

    def parse_create_process(self, event: NamedTuple) -> None:
        """Handler for Event 4688: process creation."""
        #assert event.Version == 1
        # command line wasn't enabled, use name instead
        # TODO filter out double backslashes from name
        #      NB. Doesn't actually seem to be a problem (Python will display
        #      strings as escaped, but print and the literal is correct.)
        # Actually, probably better to do this as a CharFilter in CB so it can
        # be applied to both queries and logs
        pid = int(event.EventDataFields["NewProcessId"], 16)
        ppid = int(event.EventDataFields["ProcessId"], 16)
        name = self.rewrite_path(event.EventDataFields["NewProcessName"])
        if name is None:
            name = graph.unknown_proc
        time = self.get_time(event)
        uuid = self.make_edge_uuid(event)

        src_id = self.map_pid(ppid)
        src_uuid = self.make_node_uuid(src_id)
        dst_id = self.map_pid(pid)
        dst_uuid = self.make_node_uuid(dst_id)

        node_attrs = {
            "type": "process",
            "time": time,
        }

        self.add_new_node(src_id, name=graph.unknown_proc, uuid=src_uuid, **node_attrs)
        self.add_node(dst_id, name=name, uuid=dst_uuid, **node_attrs)
        # call it "clone" just to have similar semantics to Linux logs
        self.add_edge(src_id, dst_id, time=time, type="clone", uuid=uuid)

        self.label_node(pid, name, dst_uuid)

        self.names.add(name)

    def parse_exit(self, event: NamedTuple) -> None:
        """Handler for Event 4689: process exit.

        While exits are not inherently that useful for provenance graphs, we
        need to use exits to know whether a PID is live, as Windows frequently
        reuses PIDs.
        """
        pid = int(event.EventDataFields["ProcessId"], 16)
        self.remove_pid(pid)

    def parse_access(self, event: NamedTuple) -> None:
        """Handler for Event 4663: generic object access."""
        pid = int(event.EventDataFields["ProcessId"], 16)
        proc_name = self.rewrite_path(event.EventDataFields["ProcessName"])
        if proc_name is None:
            proc_name = graph.unknown_proc
        # main object types seem to be files and keys, but while keys show up
        # in 4656, they don't show up in 4663
        # TODO how to handle directories?
        obj_type = event.EventDataFields["ObjectType"].lower()
        obj_name = self.rewrite_path(event.EventDataFields["ObjectName"])
        if obj_name is None:
            obj_name = graph.unknown_file
        time = self.get_time(event)
        uuid = self.make_edge_uuid(event)
        mask = int(event.EventDataFields["AccessMask"], 16)

        proc_id = self.map_pid(pid)
        proc_uuid = self.make_node_uuid(proc_id)
        obj_id = self.map_obj(obj_name)
        obj_uuid = self.make_node_uuid(obj_id)
        self.add_new_node(proc_id, name=proc_name, uuid=proc_uuid,
                          type="process", time=time)
        self.add_new_node(obj_id, name=obj_name, uuid=obj_uuid,
                          type=obj_type, time=time)

        self.label_node(pid, proc_name, proc_uuid)

        if mask & AccessMask.PROV_READ:
            self.add_edge(obj_id, proc_id, time=time, type="read", uuid=uuid)
        if mask & AccessMask.PROV_WRITE:
            self.add_edge(proc_id, obj_id, time=time, type="write", uuid=uuid)
        if mask & AccessMask.PROV_EXEC:
            self.add_edge(obj_id, proc_id, time=time, type="exec", uuid=uuid)

        self.names.add(proc_name)
        self.names.add(obj_name)

    @staticmethod
    def make_sock_name(addr: str, port: str) -> str:
        """Return a socket name with the given `addr` and `port`."""
        return f"sock={addr}:{port}"

    def parse_netobject_access(self, event: NamedTuple) -> None:
        """Handler for Event 5140: network share object accesses.

        Note that this is from the perspective of a remote client trying to
        access objects on this server."""
        # There's not actually ambiguity for IPv6 addresses because the port
        # is ALWAYS present. Just be sure to split the rightmost ':'
        sock_name = self.make_sock_name(event.EventDataFields["IpAddress"],
                                        event.EventDataFields["IpPort"])
        # this path's format is different (\\*\PATH)
        obj_path = event.EventDataFields["ShareLocalPath"]
        if obj_path is None:
            return
        time = self.get_time(event)
        uuid = self.make_edge_uuid(event)
        mask = int(event.EventDataFields["AccessMask"], 16)

        sock_id = self.map_obj(sock_name)
        sock_uuid = self.make_node_uuid(sock_id)
        obj_id = self.map_obj(obj_path)
        obj_uuid = self.make_node_uuid(obj_id)

        self.add_new_node(sock_id, name=sock_name, uuid=sock_uuid,
                          type="socket", time=time)
        self.add_new_node(obj_id, name=obj_path, uuid=obj_uuid,
                          type="file", time=time)

        if mask & AccessMask.PROV_READ:
            self.add_edge(obj_id, sock_id, time=time, type="read", uuid=uuid)
        if mask & AccessMask.PROV_WRITE:
            self.add_edge(sock_id, obj_id, time=time, type="write", uuid=uuid)
        if mask & AccessMask.PROV_EXEC:
            self.add_edge(obj_id, sock_id, time=time, type="exec", uuid=uuid)

        self.names.add(sock_name)
        self.names.add(obj_path)

    def parse_connect(self, event: NamedTuple) -> None:
        """Handler for Event 5156: permitted connection.

        Windows doesn't seem to log individual socket events (send/recv),
        although this might be a configuration issue (Winsock?). Regardless,
        we cannot do any better than to mark a single send or receive at the
        time of a connection."""
        try: # ATLASv1
            pid = int(event.EventDataFields["ProcessId"], 10)
        except KeyError: # ATLASv2
            pid = int(event.EventDataFields["ProcessID"], 10)

        proc_name = self.rewrite_path(event.EventDataFields["Application"])
        if proc_name is None:
            proc_name = graph.unknown_proc
        direction = event.EventDataFields["Direction"]
        sock_name = self.make_sock_name(event.EventDataFields["DestAddress"],
                                        event.EventDataFields["DestPort"])
        time = self.get_time(event)
        uuid = self.make_edge_uuid(event)

        proc_id = self.map_pid(pid)
        proc_uuid = self.make_node_uuid(proc_id)
        sock_id = self.map_obj(sock_name)
        sock_uuid = self.make_node_uuid(sock_id)

        self.add_new_node(proc_id, name=proc_name, uuid=proc_uuid,
                          type="process", time=time)
        self.add_new_node(sock_id, name=sock_name, uuid=sock_uuid,
                          type="socket", time=time)

        self.label_node(pid, proc_name, proc_uuid)

        if direction == NetDirection.IN:
            self.add_edge(sock_id, proc_id, time=time, type="recv", uuid=uuid)
        elif direction == NetDirection.OUT:
            self.add_edge(proc_id, sock_id, time=time, type="send", uuid=uuid)
        else:
            raise ValueError("Invalid direction")

    def parse_record(self, record: NamedTuple) -> None:
        if record.EventID in self.event_handlers:
            try:
                self.event_handlers[record.EventID](self, record)
            except BaseException:
                log.error("Error at record: %s", record)
                raise

    def set_current_logfile(self, path: str) -> None:
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

        # set the current label dict
        key = f"{self.host}-{typ}"
        if key in self.all_labels:
            log.info("labeling for %s", key)
            self.labels = self.all_labels[key]
        else:
            log.warning("no labels found for %s", key)
            self.labels = {}

    event_handlers = {
        4663: parse_access,
        4688: parse_create_process,
        4689: parse_exit,
        5140: parse_netobject_access,
        5156: parse_connect,
    }

def decode_node_uuid(uid: uuid.UUID) -> Tuple[str, int]:
    """Decode `uid`, generated by `make_node_uuid`, back into the node ID in
    the graph and the host."""
    host = uid.bytes[6:8].decode("utf-8")
    n_id = int.from_bytes(uid.bytes[8:], "big")
    return host, n_id

def decode_event_uuid(uid: uuid.UUID) -> Tuple[str, str, int]:
    """Decode `uid`, generated by `make_edge_uuid`, back into the host (h1 or
    h2), set_name (b?, s?, or m? for benign/singlehost attack/multihost
    attack), and edge EventRecordId."""
    typ = uid.bytes[4:6].decode("utf-8")
    host = uid.bytes[6:8].decode("utf-8")
    record_id = int.from_bytes(uid.bytes[8:], "big")
    return host, typ, record_id

def write_graph_pickle(labels_path: str, base_path: str, host: str) -> None:
    set_name = host
    basename = f"{set_name}_prov_graph"
    graph_path = f"{config.atlasv2.graphs}/{basename}.pkl.xz"
    names_path = f"{config.atlasv2.names}/{basename}.txt"
    attack_path = f"{config.atlasv2.labels}/{set_name}_attack_node_uuids.xz"
    contaminated_path = f"{config.atlasv2.labels}/{set_name}_contaminated_node_uuids.xz"
    labels = read_labels(labels_path)

    #if util.skip_overwrite(graph_path):
    #    return

    log.info("parsing %s...", basename)
    state = ParseState(host, labels)
    json_stream.parse_json_set(base_path, host, state)
    log.info("finished %s.", basename)
    #g = state.g.build()
    #log.info("  |V| = %d, |E| = %d", g.vcount(), g.ecount())

    #log.info("writing %s...", graph_path)
    #util.pickle_write(g, graph_path)

    log.info("writing %s...", attack_path)
    with lzma.open(attack_path, "wt") as f:
        for uuid in state.attack_uuids:
            f.write(str(uuid))
            f.write("\n")

    log.info("writing %s...", contaminated_path)
    with lzma.open(contaminated_path, "wt") as f:
        for uuid in state.contaminated_uuids:
            f.write(str(uuid))
            f.write("\n")

    log.info("done")

    with open(names_path, "w") as f:
        for name in state.names:
            f.write(name)
            f.write("\n")

def main(argv):
    label_path = "<PATH_TO_LABELS>/atlasv2_labels.csv"
    base = "<PATH_TO_DATASET>/atlasv2"
    hosts = ("h1", "h2")
    for host in hosts:
        write_graph_pickle(label_path, base, host)
