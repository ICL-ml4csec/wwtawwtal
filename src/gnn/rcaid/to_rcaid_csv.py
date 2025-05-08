"""Convert an `igraph` provenance graph into a CSV usable by RCAID. This uses
the following columns:

{sourcepath}!~{nodeId}, {sourceType}, {destinationPath}!~{nodeId}, {destinationType}, {syscall}, {processName}, {retTime}, {uuid}
"""

import csv
import datetime
import logging
import lzma

import igraph as ig
import tqdm

import config
import gnn.util
import util
from graph_transforms.merge_sockets import collapse_sockets

log = logging.getLogger(__name__)

def repr_node(u: ig.Vertex) -> str:
    return f"{u['name']}!~{u['nid']}"

def extract_basename(u: ig.Vertex, is_windows: bool = False) -> str:
    sep = "\\" if is_windows else "/"
    *_, name = u["name"].rsplit(sep, maxsplit=1)
    return name

def convert_graph(g: ig.Graph, out_dir: str, set_name: str,
                  is_windows: bool = False) -> None:
    fields = ("src", "srcType", "dst", "dstType", "syscall",
              "procName", "retTime", "uuid")
    with lzma.open(f"{out_dir}/{set_name}.csv.xz", "wt", newline="") as f:
    #with lzma.open(f"{out_dir}/{set_name}_relabeled_unknowns.csv.xz", "wt") as f:
    #with lzma.open(f"{out_dir}/{set_name}_unreduced.csv.xz", "wt") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        try:
            for e in tqdm.tqdm(sorted(g.es, key=lambda e: e["time"])):
                src = g.vs[e.source]
                dst = g.vs[e.target]
                # hack to get process name: look at src, then dst type
                # only clones are process-to-process, and the process should be
                # the parent. otherwise, only one of the nodes is a process.
                proctypes = {"process", "thread"}
                procName = None
                srcType = src["type"]
                dstType = dst["type"]
                if srcType in proctypes:
                    procName = extract_basename(src, is_windows)
                    # avoid confusing rcaid: just label threads as process
                    srcType = "process"
                if dst["type"] in proctypes:
                    if procName is None:
                        procName = extract_basename(dst, is_windows)
                    dstType = "process"
                if procName is None:
                    log.warning("no proc? ignoring [%s] -> [%s]",
                                src["name"], dst["name"])
                    continue
                row = {
                    "src": repr_node(src),
                    "srcType": srcType,
                    "dst": repr_node(dst),
                    "dstType": dstType,
                    "syscall": e["type"],
                    "procName": procName,
                    "retTime": e["time"],
                    "uuid": e["uuid"],
                }
                writer.writerow(row)
        except Exception:
            print("At edge: ", e)
            raise

def convert_by_name(cfg, set_name: str) -> None:
    if cfg == config.streamspot:
        graph_path = f"{cfg.graphs}/{set_name}_prov_graph.pkl.xz"
    else:
        graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    #graph_path = f"{cfg.graphs}/{set_name}_prov_graph_relabeled_unknowns_collapsed_sockets.pkl.xz"
    log.info("reading %s...", graph_path)
    g = util.pickle_read(graph_path)

    is_windows = (set_name == "fivedirections" or cfg == config.atlasv1
                  or cfg == config.atlasv2 or cfg == config.optc)

    if cfg == config.streamspot:
        g = g.g

    # add nid attribute
    g.vs["nid"] = list(range(g.vcount()))

    if cfg == config.streamspot:
        collapse_sockets(g)

    gnn.util.transform_graph(g, set_name, remove_parallel=True)

    log.info("writing csv...")
    convert_graph(g, cfg.rcaid_graphs_csv, set_name, is_windows)

def main(argv):
    #for set_name in ("cadets", "theia", "fivedirections", "trace"):
    #    convert_by_name(config.e3, set_name)
    #convert_by_name(config.e3, "trace_departitioned")
    #convert_by_name(config.atlasv2, "h1")
    #convert_by_name(config.atlasv2, "h2")
    #convert_by_name(config.atlasv2, "combined")
    #convert_by_name(config.streamspot, "all")
    for set_name in ["host201", "host501"]:
        convert_by_name(config.optc, set_name)
