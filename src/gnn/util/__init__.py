import logging

import igraph as ig

import config
import parse.graph

log = logging.getLogger(__name__)

def transform_graph(g: ig.Graph, set_name: str,
                    remove_parallel: bool = False,
                    filter_edges: bool = True) -> None:
    """Transform `g` to fit stream IDS requirements: e.g., remove unknown
    entities."""
    ### While this seems nice in theory, many unknown nodes end up being root
    ### nodes, which causes generating the streaming edges to take too long
    ### (>1000 hours).
    ## all unknown nodes:
    ## change "true unknowns" into file unknowns (they're almost always files)
    ## rename files to `/tmp/unknown`, processes to `/usr/bin/unknown`
    ## change type to `file` regardless of its original type
    #log.info("fixing unknown nodes...")
    #unknowns = g.vs(lambda u: u["name"].startswith(graph.unknown_prefix))
    ## look for execution units, change type from unknown to process
    #if set_name == "trace":
    #    for u in unknowns(type_eq="unknown"):
    #        if len(g.es(_target=u.index, type_eq="EVENT_UNIT")) > 0:
    #            u["type"] = "process"
    #unknowns(type_eq="unknown")["type"] = "file"
    #unknowns(type_eq="file")["name"] = "/tmp/unknown"
    #unknowns(type_eq="process")["name"] = "/usr/bin/unknown"
    #unknowns["type"] = "file"

    # need to address the versioned exec nodes
    # one possibility is to merge, but instead, if a parent is unknown and the
    # child is known, rename the parent to also be the child
    log.info("adjusting versioned execs with unknown parents...")
    for e in g.es(type_eq="exec"):
        src = g.vs[e.source]
        dst = g.vs[e.target]
        if (dst["name"].startswith(parse.graph.unknown_prefix)
            or dst["type"] != "process"):
            continue
        # this is actually buggy; below is the correct code
        #src["type"] = "process"
        #if src["name"].startswith(parse.graph.unknown_prefix):
        #    src["name"] = dst["name"]
        if (src["name"].startswith(parse.graph.unknown_prefix)
            and src["type"] != "file"):
            src["type"] = "process"
            src["name"] = dst["name"]

    # delete unknowns
    log.info("removing unknown nodes...")
    g.delete_vertices((u.index for u in g.vs
                       if u["name"].startswith(parse.graph.unknown_prefix)))

    # delete non-information flow events; a nice side effect is that this
    # should eliminate the connect-roots problem
    # note KAIROS and FLASH benefit greatly from having more edge types
    if filter_edges:
        desired_events = {
            "send", "sendto", "sendmsg", "recv", "recvfrom", "recvmsg",
            "clone", "exec", "loadlibrary", "read", "write"
        }
        log.info("removing unnecessary edges...")
        g.delete_edges(type_notin=desired_events)

    # for GNN, parallel edges don't matter; this is a massive reduction
    if remove_parallel:
        log.info("removing all parallel edges...")
        g.simplify(multiple=True, loops=False, combine_edges="first")
