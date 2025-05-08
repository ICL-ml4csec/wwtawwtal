import glob
import sys
import os

import networkx as nx
import parsec as ps

_NAT = ps.regex(r"\d+").parsecmap(int)
_NEG = ps.parsecmap(ps.string('-') >> _NAT, lambda n: -n)
_HEX = ps.regex(r"0x[0-9A-Fa-f]+").parsecmap(lambda h: int(h, 16))
_STR = ps.regex(r"[^,]+")
_INT = _NEG | _HEX | _NAT

def _separated_list(sep, *args):
    """Return `join(sep(args[0]), sep(args[1]), ..., sep(args[-2]), args[-1])`.
    Note that `sep` is NOT applied to the final argument."""
    return ps.joint(
        *map(sep, args[:-1]),
        args[-1]
    )

# from preprocess.py
types = { 'process': 'a',
          'thread': 'b',
          'file': 'c',
          'MAP_ANONYMOUS': 'd',
          'NA': 'e',
          'stdin': 'f',
          'stdout': 'g',
          'stderr': 'h',
          'accept': 'i',
          'access': 'j',
          'bind': 'k',
          'chmod': 'l',
          'clone': 'm',
          'close': 'n',
          'connect': 'o',
          'execve': 'p',
          'fstat': 'q',
          'ftruncate': 'r',
          'listen': 's',
          'mmap2': 't',
          'open': 'u',
          'read': 'v',
          'recv': 'w',
          'recvfrom': 'x',
          'recvmsg': 'y',
          'send': 'z',
          'sendmsg': 'A',
          'sendto': 'B',
          'stat': 'C',
          'truncate': 'D',
          'unlink': 'E',
          'waitpid': 'F',
          'write': 'G',
          'writev': 'H',
        }
types = { v: k for k, v in types.items() } # reverse the map

_type = ps.one_of("".join(types.keys()))

_src_id = _INT
_src_type = _type
_dst_id = _INT
_dst_type = _type
_edge_type = _type
_graph_id = _INT

_entry = lambda p: p << ps.string('\t')

_line = _separated_list(_entry,
    _src_id,
    _src_type,
    _dst_id,
    _dst_type,
    _edge_type,
    _graph_id,
)

def parse_file(path: str, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add the data contained in a single file to G. This expects the
    preparsed graphs of the StreamSpot dataset."""
    with open(path, "r") as f:
        for line in f:
            src_id, src_type, dst_id, dst_type, edge_type, _ = _line.parse(line)
            G.add_node(src_id, type=src_type)
            G.add_node(dst_id, type=dst_type)
            G.add_edge(src_id, dst_id, type=edge_type)
    return G

def parse_set(path: str) -> nx.MultiDiGraph:
    """Each StreamSpot dataset is broken into 100 trials. This concatenates
    all the trials together and returns a single graph."""
    G = nx.MultiDiGraph()
    for name in glob.iglob(f"{path}/*.csv"):
        # there's some other files mixed in (looks like errors?), skip them
        base = os.path.basename(name)
        base, _ = os.path.splitext(base)
        if base.isnumeric():
            print(f"\r{%s}", end="")
            parse_file(name, G)
    print("")
    return G

def main(argv) -> None:
    parse_set(argv[1])
