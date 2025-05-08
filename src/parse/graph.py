"""Graph formatting and construction for inputs to be analyzed. Although this
abstracts graph creation into adding nodes and edges, parsers for individual
log formats are still required."""

import csv
import copy
from dataclasses import dataclass # current design requires mutability
import lzma
from typing import List

import igraph as ig

@dataclass
class NodeAttr:
    type: str
    name: str
    time: int
    uuid: str

@dataclass
class EdgeAttr:
    type: str
    time: int
    uuid: str

# null byte cannot legally be in a Windows or Linux path
# Use the specific unknown_* names to ensure uniqueness. GraphBuilder will
# automatically append numbers to ensure all unknown nodes have unique names.
unknown_prefix = "\x00"
unknown_file = f"{unknown_prefix}file"
unknown_sock = f"{unknown_prefix}sock"
unknown_proc = f"{unknown_prefix}proc"

class GraphBuilder:
    """Although wrapping the underlying graph package is largely unnecessary,
    we do it here because parsing logs lends itself to incrementally adding to
    the graph, but bulk operations are significantly more performant for most
    Python graph libraries."""

    node_fields = ["type", "name", "time", "uuid"]
    edge_fields = ["type", "time", "uuid"]

    def __init__(self):
        self.node_attrs = []
        self.unknown_counts = {
            unknown_file: 0,
            unknown_sock: 0,
            unknown_proc: 0,
        }
        self.edges = []
        self.edge_attrs = [] # associated with edges

    def _name_is_unknown(self, name: str) -> str:
        return name.startswith(unknown_prefix)

    def _unique_if_unknown(self, name: str) -> str:
        """Make `name` unique if it is a unknown name."""
        if name in self.unknown_counts:
            out = name + str(self.unknown_counts[name])
            self.unknown_counts[name] += 1
            return out
        else:
            return name

    def add_node(self, id: int, type: str, name: str, time: int,
                 uuid: str, overwrite: bool = False, host: str = "") -> None:
        """Add a node with the given fields. Note that all node IDs should be
        ints increasing from 0, i.e., 0, 1, 2, etc. It is up to the parser to
        maintain this node ID mapping.

        `overwrite` controls what happens if `id` was already added. If True,
        the fields of `id` are overwritten; otherwise, they remain unchanged.
        If `name` is unknown, it never overwrites the existing name.

        If the existing name is unknown, then its name is always overwritten
        regardless of `overwrite.`

        Note that string interning should reduce the overhead of using strings
        for `type`. Ideally, the parser should ensure these type strings are
        interned (it should automatically happen if they are literals).

        The `host` field is currently ignored.
        """
        N = len(self.node_attrs)
        assert id <= N
        if id == N:
            self.node_attrs.append(NodeAttr(type=type, time=time,
                                            name=self._unique_if_unknown(name),
                                            uuid=uuid))
            return
        elif overwrite:
            self.node_attrs[id].type = type
            if not self._name_is_unknown(name):
                self.node_attrs[id].name = name
            self.node_attrs[id].time = time
            self.node_attrs[id].uuid = uuid
        else:
            if (self._name_is_unknown(self.node_attrs[id].name)
                and not self._name_is_unknown(name)):
                self.node_attrs[id].name = name
            if self.node_attrs[id].type == "unknown":
                self.node_attrs[id].type = type

    def get_node(self, node_id: int) -> NodeAttr:
        return self.node_attrs[node_id]

    def add_edge(self, src_id: int, dst_id: int, type: str, time: int,
                 uuid: str) -> None:
        """Add an edge from `src_id` to `dst_id` with the given fields.

        Note that there is currently no way to modify a previously-added edge,
        as this shouldn't be needed for provenance graphs."""
        N = len(self.node_attrs)
        assert src_id < N
        assert dst_id < N
        self.edges.append((src_id, dst_id))
        self.edge_attrs.append(EdgeAttr(type=type, time=time, uuid=uuid))

    def build(self) -> ig.Graph:
        """Build the graph with all added nodes and edges."""
        vertex_attrs = {field: [n.__dict__[field] for n in self.node_attrs]
                        for field in self.node_fields}
        edge_attrs = {field: [e.__dict__[field] for e in self.edge_attrs]
                      for field in self.edge_fields}
        return ig.Graph(
            n=len(self.node_attrs),
            edges=self.edges,
            directed=True,
            vertex_attrs=vertex_attrs,
            edge_attrs=edge_attrs,
        )

    def write_csv(self, out_prefix: str) -> None:
        """Save the graph as a pair of CSVs to be rebuilt later; one file is
        `prefix_path`.nodes.csv.xz containing the node data and the other is
        `prefix_path`.edges.csv.xz containing the edge data.

        This method cannot be used to save a partial graph. The written graph
        can only be read into a complete graph, not a GraphBuilder.

        Note that while this and `read_csv` have been lightly tested, they are
        rarely used because pickled graphs read and write significantly faster
        than this ad-hoc serialization."""
        node_path = f"{out_prefix}.nodes.csv.xz"
        edge_path = f"{out_prefix}.edges.csv.xz"

        with lzma.open(node_path, "wt", newline="") as f:
            writer = csv.DictWriter(f, self.node_fields)
            writer.writeheader()
            for attr in self.node_attrs:
                # escape leading NUL bytes, CSV reader can't handle them
                # There shouldn't be any NUL bytes anywhere else since they're
                # not legal names
                row = copy.deepcopy(attr.__dict__)
                if row["name"].startswith(unknown_prefix):
                    row["name"] = row["name"].replace(unknown_prefix, "\\0", 1)
                elif row["name"].startswith("\\"):
                    # escape leading backslash to avoid mangling
                    row["name"] = "\\" + row["name"]
                writer.writerow(row)

        with lzma.open(edge_path, "wt", newline="") as f:
            writer = csv.DictWriter(f, ["src", "dst"] + self.edge_fields)
            writer.writeheader()
            for ((src, dst), attr) in zip(self.edges, self.edge_attrs):
                row = copy.deepcopy(attr.__dict__)
                row["src"] = src
                row["dst"] = dst
                writer.writerow(row)

    def write_unified_csv(self, out_prefix: str) -> None:
        """Save the graph as a single CSV in the following format:

        src_name,src_type,dst_name,dst_type,syscall,time

        This method omits all unknown names entirely.
        """
        csv_path = f"{out_prefix}.csv.xz"
        fmt = ("src_name", "src_type", "dst_name", "dst_type", "syscall", "time_ns")

        with lzma.open(csv_path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fmt)
            writer.writeheader()
            for ((src_id, dst_id), edge_attr) in zip(self.edges, self.edge_attrs):
                src_attr = self.node_attrs[src_id]
                dst_attr = self.node_attrs[dst_id]
                if (src_attr.name.startswith(unknown_prefix) or
                    dst_attr.name.startswith(unknown_prefix)):
                    continue
                row = {
                    fmt[0]: src_attr.name,
                    fmt[1]: src_attr.type,
                    fmt[2]: dst_attr.name,
                    fmt[3]: dst_attr.type,
                    fmt[4]: edge_attr.type,
                    fmt[5]: edge_attr.time,
                }
                writer.writerow(row)

    @staticmethod
    def read_csv(csv_prefix: str) -> ig.Graph:
        """Build a graph that was saved by `write_csv`."""
        node_path = f"{csv_prefix}.nodes.csv.xz"
        edge_path = f"{csv_prefix}.edges.csv.xz"

        builder = GraphBuilder()
        with lzma.open(node_path, "rt", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = int(row.pop("time"))
                # unescape leading NUL bytes or backslashes now
                if row["name"].startswith("\\"):
                    if row["name"][1] == "0":
                        row["name"] = row["name"].replace("\\0", unknown_prefix, 1)
                    else:
                        row["name"] = row["name"][1:]
                builder.node_attrs.append(NodeAttr(time=time, **row))

        with lzma.open(edge_path, "rt", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = int(row.pop("src"))
                dst = int(row.pop("dst"))
                time = int(row.pop("time"))
                builder.edges.append((src, dst))
                builder.edge_attrs.append(EdgeAttr(time=time, **row))

        return builder.build()
