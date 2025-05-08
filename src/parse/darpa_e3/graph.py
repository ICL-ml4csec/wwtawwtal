"""Parses DARPA TC3 JSON data into provenance graphs/process trees."""

import enum
import logging
import shlex
from typing import Dict, List, Optional, TextIO, Tuple

import config
from parse import graph
from parse.darpa_e3 import json_stream
import util

log = logging.getLogger(__name__)

@enum.unique
class GraphType(enum.Enum):
    PROC_TREE = enum.auto()
    PROV_GRAPH = enum.auto()

class ParseState(json_stream.AbstractParseState):
    def set_data_handlers(self):
        if self.graph_type == GraphType.PROV_GRAPH:
            self.data_handlers = {
                f"{self.ns}Event": self.parse_event,
                f"{self.ns}FileObject": self.parse_fileobject,
                f"{self.ns}NetFlowObject": self.parse_netflowobject,
                f"{self.ns}RegistryKeyObject": self.parse_registrykeyobject,
                f"{self.ns}Subject": self.parse_subject,
            }
            self.event_handlers = {
                "EVENT_CLONE": self.parse_clone,
                "EVENT_FORK": self.parse_fork,
                "EVENT_EXECUTE": self.parse_execute,
                "EVENT_READ": self.parse_read,
                "EVENT_WRITE": self.parse_write,
                "EVENT_MMAP": self.parse_mmap,
                #"EVENT_RENAME": self.parse_rename,
                #"EVENT_LINK": self.parse_link,
                "EVENT_OPEN": self.parse_open,
                "EVENT_SENDMSG": self.parse_sendmsg,
                "EVENT_SENDTO": self.parse_sendto,
                "EVENT_RECVMSG": self.parse_recvmsg,
                "EVENT_RECVFROM": self.parse_recvfrom,
                "EVENT_LOADLIBRARY": self.parse_loadlibrary,
            }
            self.use_default_handlers = True
        elif self.graph_type == GraphType.PROC_TREE:
            self.data_handlers = {
                f"{self.ns}Subject": self.parse_subject,
            }
            self.event_handlers = {}
            if self.set_name == "cadets":
                # CADETS Subject records have no name, need to parse execute
                self.data_handlers[f"{self.ns}Event"] = self.parse_event
                self.event_handlers["EVENT_EXECUTE"] = self.parse_execute
            self.use_default_handlers = False
        else:
            raise ValueError("invalid graph type")

    def __init__(self, set_name: str, typ: GraphType):
        super().__init__(set_name)
        # Avoid parallel edges, it costs too much memory
        # (some loss due to approximation)
        # Still use MultiDiGraph in case multi edges are wanted
        self.g = graph.GraphBuilder()
        self.node_ids = {} # host UUID -> subj/obj UUID -> int
        self.next_id = 0
        self.path_fake_uuid_map = {} # path -> fake uuid
        self.fake_uuid = 0
        self.prev_time = 0 # timestamp of the previously-added node or edge
        self.doc2vec_names = set()
        self.set_name = set_name

        self.graph_type = typ
        self.set_data_handlers()

    def next_archive(self) -> None:
        super().next_archive()
        self.set_data_handlers() # reset these, since they depend on self.ns

    def map_id(self, host_uuid: str, uuid: str) -> int:
        """Map the subject/object UUID to an int, and return the mapped id."""
        if host_uuid not in self.node_ids:
            self.node_ids[host_uuid] = {}
        host_map = self.node_ids[host_uuid]
        if uuid not in host_map:
            host_map[uuid] = self.next_id
            self.next_id += 1
        return host_map[uuid]

    def remap_id(self, host_uuid: str, uuid: str) -> Tuple[Optional[int], int]:
        """Always map `uuid` to a new int, and return the old id (if it
        exists) and new mapped id. This is useful for versioning."""
        if host_uuid not in self.node_ids:
            self.node_ids[host_uuid] = {}
        host_map = self.node_ids[host_uuid]
        old_id = host_map.get(uuid)
        host_map[uuid] = self.next_id
        self.next_id += 1
        return old_id, host_map[uuid]

    def add_new_node(self, node: int, time: int, **kwargs):
        """Add `node` to `self.g` only if it does not already exist."""
        self.prev_time = time
        self.g.add_node(node, time=time, overwrite=False, **kwargs)

    def add_node(self, node: int, time: int, **kwargs):
        """Add `node` to `self.g`."""
        self.prev_time = time
        self.g.add_node(node, time=time, overwrite=True, **kwargs)

    def add_edge(self, src: int, dst: int, **kwargs):
        """Add an edge from `src` to `dst` in `self.g`."""
        # Avoiding parallel edges doesn't seem to really help much with memory
        # consumption; the problem likely stems elsewhere
        self.g.add_edge(src, dst, **kwargs)

    # clone/fork doesn't have any info about the program path, etc., so use
    # SUBJECT to get those, and use clone/fork only to get the event UUID
    def parse_clone(self, event: Dict) -> None:
        """Parse a fork/clone event. There is a separate EVENT_CREATE_THREAD
        event, so EVENT_CLONE is most likely for new processes instead of new
        threads.

        Depending on the dataset, clone events may or may not actually have
        process name info. For example, CADETS does have the forked process
        name, whereas THEIA does not."""
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        src_uuid = event["subject"][f"{self.ns}UUID"]
        src_id = self.map_id(host, src_uuid)
        dst_uuid = event["predicateObject"][f"{self.ns}UUID"]
        dst_id = self.map_id(host, dst_uuid)

        node_attrs = {
            "type": "process",
            "time": time,
            "host": host
        }

        # in CADETS, __all__ events have the "exec" field which seems to
        # indicate the current process name. This will correctly set the
        # forked process name, unless an exec later changes it.
        if self.set_name == "cadets":
            try:
                proc_name = event["properties"]["map"]["exec"]
            except (KeyError, TypeError):
                proc_name = graph.unknown_proc
        else:
            proc_name = graph.unknown_proc
        # N.B.: add_new_node __will__ overwrite the name if it was unknown
        self.add_new_node(src_id, name=proc_name, uuid=src_uuid, **node_attrs)
        self.add_new_node(dst_id, name=proc_name, uuid=dst_uuid, **node_attrs)

        self.doc2vec_names.add(proc_name)

        self.add_edge(src_id, dst_id, type="clone", time=time, uuid=uuid)

    # Linux uses EVENT_CLONE, but BSD/Windows uses EVENT_FORK
    def parse_fork(self, event: Dict) -> None:
        # reuse clone logic
        self.parse_clone(event)

    def parse_execute(self, event: Dict) -> None:
        """Parse a PARSE_EXECUTE event to learn the correct process name. This
        overwrites a node's ID, although technically, it is possible that exec
        could represent two distinct processes."""
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]
        proc_uuid = event["subject"][f"{self.ns}UUID"]

        try:
            proc_name = event["predicateObjectPath"]["string"]
        except (KeyError, TypeError):
            try:
                cmdline = event["properties"]["map"]["cmdLine"]
                proc_name, *_ = shlex.split(cmdline)
            except ValueError:
                proc_name, *_ = cmdline.split(maxsplit=1)
            except KeyError:
                proc_name = graph.unknown_proc

        bin_uuid = event["predicateObject"][f"{self.ns}UUID"]
        bin_id = self.map_id(host, bin_uuid)
        self.add_new_node(bin_id, host=host, name=proc_name, type="file",
                          time=time, uuid=bin_uuid)

        # version the node by using remap. although 99% of the time, exec is a
        # fork -> exec, there could be situations where this isn't true
        # (especially during attacks), so we want to know what became what.
        old_proc_id, proc_id = self.remap_id(host, proc_uuid)
        self.add_node(proc_id, host=host, name=proc_name, type="process",
                      time=time, uuid=proc_uuid)
        if old_proc_id is not None:
            self.add_new_node(old_proc_id, host=host, name=proc_name,
                              type="process", time=time, uuid=proc_uuid)
            self.add_edge(old_proc_id, proc_id, type="exec", time=time, uuid=uuid)

        # for some reason TRACE sometimes uses exec instead of clone
        # we checked this for all other datasets (except ClearScope, which we
        # don't use due to its timestamp issue)
        if self.set_name == "trace":
            # "binary file" is actually a process
            if self.g.node_attrs[bin_id].type == "process":
                self.add_edge(proc_id, bin_id, type="clone", time=time, uuid=uuid)
            elif self.graph_type != GraphType.PROC_TREE:
                self.add_edge(bin_id, proc_id, type="exec", time=time, uuid=uuid)
        else:
            # create an exec edge if not process tree
            if self.graph_type != GraphType.PROC_TREE:
                self.add_edge(bin_id, proc_id, type="exec", time=time, uuid=uuid)

        self.doc2vec_names.add(proc_name)

    def parse_loadlibrary(self, event: Dict) -> None:
        """Treat LOADLIBRARY as an exec, but note that they don't have
        corresponding FILE objects, so this must be handled specially."""
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        lib_name = event["predicateObjectPath"]["string"]
        lib_uuid = event["predicateObject"][f"{self.ns}UUID"]
        lib_id = self.map_id(host, lib_uuid)

        # possibly the lib's name, as the FileObject record doesn't have it
        self.add_new_node(lib_id, host=host, name=lib_name, type="file",
                          time=time, uuid=lib_uuid)
        self.doc2vec_names.add(lib_name)

        proc_uuid = event["subject"][f"{self.ns}UUID"]
        proc_id = self.map_id(host, proc_uuid)
        self.add_new_node(proc_id, host=host, name=graph.unknown_proc,
                          uuid=proc_uuid, time=time, type="process")

        self.add_edge(lib_id, proc_id, type="loadlibrary", time=time, uuid=uuid)

    def parse_mmap(self, event: Dict) -> None:
        """Adds a mmap event to the provenance graph as either a read, write,
        or both (if both, add two edges going in either direction).

        This has to do a fair amount of approximation because the CDM format
        of MMAP is not very stable across the different TC datasets."""
        # CADETS: protection is a string of C constants. object1 is file
        # THEIA: protection is a string of C constants. MMAP might actually be
        # an munmap event. object1 is memory object (ignore), object2 is file
        # TRACE: protection is an int; seems to always be an MMAP. object1 is
        # file, object2 is memory object (ignore)
        # Assume Linux semantics:
        #   READ = 1, WRITE = 2, EXEC = 4
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        dirs = []
        if self.set_name == "theia":
            if event["name"] != "mmap":
                return # ignore anything that isn't mmap (e.g., munmap)
            if "PROT_READ" in event["properties"]["map"]["prot"]:
                dirs.append("read")
            if "PROT_WRITE" in event["properties"]["map"]["prot"]:
                dirs.append("write")
            file_uuid = event["predicateObject2"][f"{self.ns}UUID"]
            file_name = graph.unknown_file
        elif self.set_name == "cadets":
            if "PROT_READ" in event["properties"]["map"]["arg_mem_flags"]:
                dirs.append("read")
            if "PROT_WRITE" in event["properties"]["map"]["arg_mem_flags"]:
                dirs.append("write")
            file_uuid = event["predicateObject"][f"{self.ns}UUID"]
            file_name = event["properties"]["map"]["partial_path"]
        elif self.set_name == "trace":
            PROT_READ = 0x1
            PROT_WRITE = 0x2
            prot = int(event["properties"]["map"]["protection"])
            if (PROT_READ & prot) != 0:
                dirs.append("read")
            if (PROT_WRITE & prot) != 0:
                dirs.append("write")
            file_uuid = event["predicateObject"][f"{self.ns}UUID"]
            try:
                file_name = event["predicateObjectPath"]["string"]
            except (KeyError, TypeError):
                file_name = graph.unknown_file
        else:
            raise ValueError(f"cannot handle mmap for set {self.set_name}")

        proc_uuid = event["subject"][f"{self.ns}UUID"]
        node_attrs = {
            "time": time,
            "host": host,
        }

        file_id = self.map_id(host, file_uuid)
        self.add_new_node(file_id, name=file_name, uuid=file_uuid,
                          type="file", **node_attrs)

        proc_id = self.map_id(host, proc_uuid)
        self.add_new_node(proc_id, name=graph.unknown_proc, uuid=proc_uuid,
                          type="process", **node_attrs)
        if "read" in dirs:
            self.add_edge(file_id, proc_id, type="mmap", time=time, uuid=uuid)
        if "write" in dirs:
            self.add_edge(proc_id, file_id, type="mmap", time=time, uuid=uuid)

        self.doc2vec_names.add(file_name)

    def _handle_file_io(self, event: Dict, typ: str) -> None:
        """Adds a read/write to the provenance graph. (Assume read/write only
        do file I/O, although this is not necessarily true.)"""
        # use preprocessed data, it seems reliable enough
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        try:
            file_name = event["predicateObjectPath"]["string"]
            # substitute Cadets unknown name for our own
            if self.set_name == "cadets" and file_name == "<unknown>":
                file_name = graph.unknown_file
        except (KeyError, TypeError):
            file_name = graph.unknown_file

        node_attrs = {
            "type": "file",
            "time": time,
            "host": host,
        }

        # cadets files are occasionally missing UUIDs, use a fake one
        try:
            file_uuid = event["predicateObject"][f"{self.ns}UUID"]
        except (KeyError, TypeError):
            if file_name.startswith(graph.unknown_prefix):
                file_uuid = self.fake_uuid
                self.fake_uuid += 1
            else:
                if file_name not in self.path_fake_uuid_map:
                    self.path_fake_uuid_map[file_name] = self.fake_uuid
                    self.fake_uuid += 1
                file_uuid = self.path_fake_uuid_map[file_name]
        file_id = self.map_id(host, file_uuid)
        self.add_new_node(file_id, name=file_name, uuid=file_uuid, **node_attrs)

        node_attrs["type"] = "process"
        proc_uuid = event["subject"][f"{self.ns}UUID"]
        proc_id = self.map_id(host, proc_uuid)
        self.add_new_node(proc_id, name=graph.unknown_proc, uuid=proc_uuid, **node_attrs)

        self.doc2vec_names.add(file_name)
        if typ == "read" or typ == "open":
            # technically, open MIGHT write (in a truncating open)
            # but handling open correctly requires handling the flags for
            # every possible OS, since they are not all the same
            self.add_edge(file_id, proc_id, type=typ, time=time, uuid=uuid)
        elif typ == "write":
            self.add_edge(proc_id, file_id, type=typ, time=time, uuid=uuid)
        else:
            raise ValueError("invalid type")

    def parse_read(self, event: Dict) -> None:
        self._handle_file_io(event, "read")

    def parse_write(self, event: Dict) -> None:
        self._handle_file_io(event, "write")

    def parse_open(self, event: Dict) -> None:
        """Parse the `open` event. This seems mostly unnecessary, but CADETS
        doesn't include file name information in its FileObject records, so we
        need to track OPEN events to see it instead."""
        self._handle_file_io(event, "open")

    # causality: this looks like proc writes object2 (dst)
    def parse_rename(self, event: Dict) -> None:
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        node_attrs = {
            "type": "file",
            "time": time,
            "host": host,
        }

        #try:
        #    src_name = event["predicateObjectPath"]["string"]
        #except (KeyError, TypeError):
        #    src_name = graph.unknown_file
        #src_uuid = event["predicateObject"][f"{self.ns}UUID"]
        #src_id = self.map_id(host, src_uuid)
        #self.add_new_node(src_id, name=src_name, uuid=src_uuid, **node_attrs)

        try:
            dst_name = event["predicateObject2Path"]["string"]
        except (KeyError, TypeError):
            dst_name = graph.unknown_file
        dst_uuid = event["predicateObject2"][f"{self.ns}UUID"]
        dst_id = self.map_id(host, dst_uuid)
        self.add_new_node(dst_id, name=dst_name, uuid=dst_uuid, **node_attrs)

        node_attrs["type"] = "process"
        proc_uuid = event["subject"][f"{self.ns}UUID"]
        proc_id = self.map_id(host, proc_uuid)
        self.add_new_node(proc_id, name=graph.unknown_proc, uuid=proc_uuid, **node_attrs)

        #self.doc2vec_names.add(src_name)
        self.doc2vec_names.add(dst_name)
        self.add_edge(proc_id, dst_id, type="rename", time=time, uuid=uuid)

    # same logic as parse_rename
    def parse_link(self, event: Dict) -> None:
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        node_attrs = {
            "type": "file",
            "time": time,
            "host": host,
        }

        try:
            dst_name = event["predicateObject2Path"]["string"]
        except (KeyError, TypeError):
            dst_name = graph.unknown_file
        dst_uuid = event["predicateObject2"][f"{self.ns}UUID"]
        dst_id = self.map_id(host, dst_uuid)
        self.add_new_node(dst_id, name=dst_name, uuid=dst_uuid, **node_attrs)

        node_attrs["type"] = "process"
        proc_uuid = event["subject"][f"{self.ns}UUID"]
        proc_id = self.map_id(host, proc_uuid)
        self.add_new_node(proc_id, name=graph.unknown_proc, uuid=proc_uuid, **node_attrs)

        self.doc2vec_names.add(dst_name)
        self.add_edge(proc_id, dst_id, type="link", time=time, uuid=uuid)

    def _handle_socket_io(self, event: Dict, typ: str) -> None:
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        node_attrs = {
            "type": "socket",
            "time": time,
            "host": host,
        }

        try:
            sock_name = event["predicateObjectPath"]["string"]
        except (KeyError, TypeError):
            sock_name = graph.unknown_sock
        sock_uuid = event["predicateObject"][f"{self.ns}UUID"]
        sock_id = self.map_id(host, sock_uuid)
        self.add_new_node(sock_id, name=sock_name, uuid=sock_uuid, **node_attrs)

        node_attrs["type"] = "process"
        proc_uuid = event["subject"][f"{self.ns}UUID"]
        proc_id = self.map_id(host, proc_uuid)
        self.add_new_node(proc_id, name=graph.unknown_proc, uuid=proc_uuid, **node_attrs)

        self.doc2vec_names.add(sock_name)
        if typ == "recvfrom" or typ == "recvmsg":
            self.add_edge(sock_id, proc_id, type=typ, time=time, uuid=uuid)
        elif typ == "sendto" or typ == "sendmsg":
            self.add_edge(proc_id, sock_id, type=typ, time=time, uuid=uuid)
        else:
            raise ValueError("invalid type")

    def parse_sendmsg(self, event: Dict) -> None:
        self._handle_socket_io(event, "sendmsg")

    def parse_sendto(self, event: Dict) -> None:
        self._handle_socket_io(event, "sendto")

    def parse_recvmsg(self, event: Dict) -> None:
        self._handle_socket_io(event, "recvmsg")

    def parse_recvfrom(self, event: Dict) -> None:
        self._handle_socket_io(event, "recvfrom")

    # We don't actually need to keep all the events. A better approach is to
    # do a trace on our ordinary provenance graph, then collect events
    # involving the implicated processes
    def default_event_handler(self, event: Dict) -> None:
        """Add a generic event to the graph for labeling."""
        # skip some always-benign and/or non-iflow events
        typ = event["type"]
        host = event["hostId"]
        time = event["timestampNanos"]
        uuid = event["uuid"]

        backwards_events = {
            # many of these are ambiguous (may read or write), but the call
            # itself usually won't write (except open, which could truncate a
            # file on write). thus, treat as reads
            #"EVENT_OPEN",
            "EVENT_CLOSE",
            #"EVENT_MPROTECT",
            #"EVENT_SHM",
            "EVENT_CONNECT",
            #"EVENT_BIND",
            "EVENT_ACCEPT",
            #"EVENT_READ_SOCKET_PARAMS",
        }

        forwards_events = {
            "EVENT_CREATE_THREAD",
            "EVENT_CREATE_OBJECT", # not exactly sure what is being created
            #"EVENT_WRITE_SOCKET_PARAMS",
            "EVENT_UNLINK",
            #"EVENT_MODIFY_FILE_ATTRIBUTES",
            #"EVENT_CHANGE_PRINCIPAL", # not sure about this one
            "EVENT_UNIT",
            "EVENT_TRUNCATE",
            "EVENT_UPDATE",
        }

        if typ in forwards_events:
            is_forward = True
        elif typ in backwards_events:
            is_forward = False
        else:
            return # just ignore missing events
            #raise KeyError(event)

        proc_uuid = event["subject"][f"{self.ns}UUID"]
        proc_id = self.map_id(host, proc_uuid)

        try:
            obj_uuid = event["predicateObject"][f"{self.ns}UUID"]
        except (KeyError, TypeError):
            # cadets files are occasionally missing UUIDs, use a fake one
            if self.set_name == "cadets":
                try:
                    file_name = event["predicateObjectPath"]["string"]
                    # substitute Cadets unknown name for our own
                    if file_name == "<unknown>":
                        file_name = graph.unknown_file
                except (KeyError, TypeError):
                    file_name = graph.unknown_file

                if file_name.startswith(graph.unknown_prefix):
                    obj_uuid = self.fake_uuid
                    self.fake_uuid += 1
                else:
                    if file_name not in self.path_fake_uuid_map:
                        self.path_fake_uuid_map[file_name] = self.fake_uuid
                        self.fake_uuid += 1
                    obj_uuid = self.path_fake_uuid_map[file_name]
            elif self.set_name == "fivedirections":
                # fivedirections might have a mysterious targetObject instead
                # of a file UUID
                obj_uuid = event["properties"]["map"]["targetObject"]
            else:
                raise
        obj_id = self.map_id(host, obj_uuid)

        self.add_new_node(proc_id, name=graph.unknown_proc, type="process",
                          host=host, time=time, uuid=proc_uuid)
        # maybe not actually a file, but pretend it is
        self.add_new_node(obj_id, name=graph.unknown_file, type="unknown",
                          host=host, time=time, uuid=obj_uuid)

        if is_forward:
            self.add_edge(proc_id, obj_id, type=typ, time=time, uuid=uuid)
        else:
            self.add_edge(obj_id, proc_id, type=typ, time=time, uuid=uuid)

    def parse_event(self, event: Dict) -> None:
        """Add the data of `event` to the provenance graph."""
        # unwrap the event, we don't need the metadata
        if event["type"] in self.event_handlers:
            self.event_handlers[event["type"]](event)
        elif self.use_default_handlers:
            self.default_event_handler(event)

    def parse_subject(self, subject: Dict) -> None:
        """Add processes in `subject` to the provenance graph."""
        host = subject["hostId"]
        time = subject["startTimestampNanos"]
        uuid = subject["uuid"]
        # if src_id == dst_id, this is probably init (with UUID of all 0s)
        # this case doesn't actually need to be handled differently
        # note that the timestamp of init is erroneously reported to be 0
        #
        # add src_id if it's unknown, but don't overwrite
        # unfortunately, the subject logs are not in temporal order, so node 0
        # is not necessarily init

        node_attrs = {
            "type": "process",
            "time": time, # use child time as default
            "host": host,
        }

        if subject["parentSubject"] is None:
            # no parent, just add this node
            n_id = self.map_id(host, uuid)
            self.add_new_node(n_id, name=graph.unknown_proc, uuid=uuid, **node_attrs)
            return

        src_uuid = subject["parentSubject"][f"{self.ns}UUID"]
        src_id = self.map_id(host, src_uuid)
        self.add_new_node(src_id, name=graph.unknown_proc, uuid=src_uuid, **node_attrs)
        # always overwrite dst_id
        # NOTE: not sure if DARPA correctly handles exec as multiple processes
        #
        # for name, path and cmdLine don't always agree, but prefer path
        # because it resolves PATH expansion
        try:
            if self.set_name == "fivedirections":
                if subject["type"] == "SUBJECT_THREAD":
                    # thread has no name, so inherit from parent
                    dst_name = self.g.get_node(src_id).name
                    # thought about maybe adding thread info somewhere (either
                    # in name or type field), but would either mess up
                    # embeddings in name or code in type
                else:
                    # FiveDirections only has cmdLine
                    try:
                        dst_name, *_ = shlex.split(subject["cmdLine"]["string"])
                    except ValueError: # encoded payloads are often cut off
                        dst_name, _ = subject["cmdLine"]["string"].split(maxsplit=1)
            else:
                # for other datasets, if no cmdLine then there is no name
                try:
                    dst_name = subject["properties"]["map"]["path"]
                except (KeyError, TypeError):
                    dst_name = subject["properties"]["map"]["name"]
        except (KeyError, TypeError):
            dst_name = graph.unknown_proc
        dst_id = self.map_id(host, uuid)
        self.add_node(dst_id, name=dst_name, uuid=uuid, **node_attrs)

        self.doc2vec_names.add(dst_name)
        # don't add the CLONE event here, we need to know its UUID
        #self.add_edge(src_id, dst_id, type="clone", time=time)

    def parse_fileobject(self, obj: Dict) -> None:
        uuid = obj["uuid"]
        host = obj["baseObject"]["hostId"]
        try:
            name = obj["baseObject"]["properties"]["map"]["path"]
        except (KeyError, TypeError):
            name = graph.unknown_file

        file_id = self.map_id(host, uuid)

        self.add_node(file_id, name=name, uuid=uuid, type="file", host=host,
                      time=self.prev_time)
        self.doc2vec_names.add(name)

    def parse_netflowobject(self, obj: Dict) -> None:
        """Parse a `NetFlowObject`, which is essentially a network socket."""
        uuid = obj["uuid"]
        host = obj["baseObject"]["hostId"]

        node_attrs = {
            "type": "socket",
            "host": host,
            "time": self.prev_time, # unknown; approximate as previous time
        }

        sock_name = f"sock={obj['remoteAddress']}:{obj['remotePort']}"
        sock_id = self.map_id(host, uuid)
        self.add_node(sock_id, name=sock_name, uuid=uuid, **node_attrs)

    def parse_registrykeyobject(self, obj: Dict) -> None:
        uuid = obj["uuid"]
        host = obj["baseObject"]["hostId"]

        node_attrs = {
            "type": "regkey",
            "host": host,
            "time": self.prev_time, # unknown; approximate as previous time
        }

        key_name = obj["key"]
        key_id = self.map_id(host, uuid)
        self.add_node(key_id, name=key_name, uuid=uuid, **node_attrs)
        self.doc2vec_names.add(key_name)

    def default_object_handler(self, typ: str, obj: Dict) -> None:
        """Parse a generic Object."""
        uuid = obj["uuid"]
        host = obj["baseObject"]["hostId"]

        node_attrs = {
            "type": typ,
            "host": host,
            "time": self.prev_time, # unknown; approximate as previous time
        }

        obj_id = self.map_id(host, uuid)
        self.add_new_node(obj_id, name=graph.unknown_file, uuid=uuid, **node_attrs)

    def parse_record(self, record: Dict) -> None:
        try:
            for k, v in record["datum"].items():
                if k in self.data_handlers:
                    self.data_handlers[k](v)
                elif k == f"{self.ns}Host" or f"{self.ns}Principal":
                    return # skip Host record
                elif self.use_default_handlers:
                    self.default_object_handler(k, v)
                # datum should only have one field
                return
        except BaseException:
           log.error("Error at record: %s", record)
           raise

def pickle_write_graph(tar_dir: str, typ: GraphType) -> None:
    """Construct the graph for `tar_dir` and write it to intermediates."""
    set_name = tar_dir.rsplit("/", maxsplit=1)[1].split("-", maxsplit=1)[0]
    basename = f"{set_name}_{typ.name.lower()}"
    graph_path = f"{config.e3.graphs}/{basename}.pkl.xz"
    names_path = f"{config.e3.names}/{basename}.txt"

    # check first since these files are super big
    if util.skip_overwrite(graph_path):
        return

    log.info("parsing %s...", basename)
    state = ParseState(set_name, typ)
    json_stream.parse_dataset(tar_dir, state)
    log.info("finished %s.", basename)

    log.info("writing all names to %s", names_path)
    with open(names_path, "w") as f:
        for name in state.doc2vec_names:
            f.write(name)
            f.write("\n")

    log.info("building provenance graph")
    g = state.g.build()
    log.info("  |V| = %d, |E| = %d", g.vcount(), g.ecount())
    log.info("  known hosts: %d", len(state.node_ids))

    del state # free up memory, pickle can be slow and use lots of memory

    log.info("writing %s...", graph_path)
    util.pickle_write(g, graph_path)
    log.info("done")

def write_graph_csv(tar_dir: str, typ: GraphType) -> None:
    """Construct the graph for `tar_dir` and write it as a CSV to
    intermediates."""
    set_name = util.tc_set_name(tar_dir)
    basename = f"{set_name}_{typ.name.lower()}"
    graph_path = f"{config.e3.graphs}/{basename}"
    names_path = f"{config.e3.names}/{basename}.txt"

    # check first since these files are super big
    if util.skip_overwrite(f"{graph_path}.nodes.csv.xz"):
        return

    log.info("parsing %s...", basename)
    state = ParseState(set_name, typ)
    json_stream.parse_dataset(tar_dir, state)
    log.info("saving %s...", basename)
    g = state.g.write_unified_csv(graph_path)
    log.info("done")

    with open(names_path, "w") as f:
        for name in state.doc2vec_names:
            f.write(name)
            f.write("\n")

def main(argv: List[str]) -> None:
    basedir = "<PATH_TO_DARPA_E3>"
    set_names = ["cadets-complete", "theia-complete", "fivedirections-complete", "trace"]
    for set_name in set_names:
        path = f"{basedir}/{set_name}"
        #pickle_write_graph(path, GraphType.PROC_TREE)
        pickle_write_graph(path, GraphType.PROV_GRAPH)
        #write_graph_csv(path, GraphType.PROV_GRAPH)
