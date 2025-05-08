"""Various ATLASv2 utilities, such as extracting benign subgraphs or
truncating off startup and attack periods."""

import datetime
import logging

import igraph as ig
import numpy as np
import tqdm

import config
import util

log = logging.getLogger(__name__)

def truncate_graph(g: ig.Graph) -> None:
    """Truncate `g` in-place, such that all data from the attack period is
    removed.

    This simply eliminates all data starting from 2022-7-19, which is the
    first day of the attack period."""
    cutoff = int(datetime.datetime(2022, 7, 19).timestamp())
    cutoff *= int(1e9)

    log.info("removing all nodes and edges with time > %s...", cutoff)
    g.delete_vertices(g.vs.select(time_ge=cutoff))
    g.delete_edges(g.es.select(time_ge=cutoff))

def write_benign_graph(path: str) -> None:
    """Given `path` to an atlasv2 graph, write out its benign subgraph."""
    directory, basename = path.rsplit("/", maxsplit=1)
    basename, _ = basename.split(".", maxsplit=1)
    host, graph_type = basename.split("_", maxsplit=1)
    out_path = f"{directory}/{host}_benign_{graph_type}.pkl.xz"

    if util.skip_overwrite(out_path):
        return

    log.info("reading %s...", path)
    g = util.pickle_read(path)
    truncate_graph(g)

    log.info("writing %s...", out_path)
    util.pickle_write(g, out_path)

def make_atlasv2_benign() -> None:
    for host in ("h1", "h2"):
        path = f"{config.atlasv2.graphs}/{host}_prov_graph.pkl.xz"
        write_benign_graph(path)

def main(argv):
    make_atlasv2_benign()
