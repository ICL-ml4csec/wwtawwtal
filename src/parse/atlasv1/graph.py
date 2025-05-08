import logging

import config
import parse.atlasv1.json_stream
import parse.atlasv2.graph
import util

log = logging.getLogger(__name__)

def write_graph_pickle(host: str) -> None:
    basename = f"{host}_prov_graph"
    graph_path = f"{config.atlasv1.graphs}/{basename}.pkl.xz"
    names_path = f"{config.atlasv1.names}/{basename}.txt"

    if util.skip_overwrite(graph_path):
        return

    log.info("parsing %s...", basename)
    state = parse.atlasv2.graph.ParseState(host)
    parse.atlasv1.json_stream.parse_json_set(host, state)
    log.info("finished %s.", basename)
    g = state.g.build()
    log.info("  |V| = %d, |E| = %d", g.vcount(), g.ecount())

    log.info("writing %s...", graph_path)
    util.pickle_write(g, graph_path)
    log.info("done")

    with open(names_path, "w") as f:
        for name in state.names:
            f.write(name)
            f.write("\n")

def main(argv):
    for arg in argv[1:]:
        write_graph_pickle(arg)
