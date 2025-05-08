"""Some of the old attack graphs were generated from provenance graphs with
some parsing errors, so they are missing process name data. This reads the new
provenance graphs and replaces the attack subgraph node names with their
corrected versions."""

import glob
import logging

import igraph as ig
import tqdm

import config
import parse.graph
import util

log = logging.getLogger(__name__)

def relabel_attack(atk: ig.Graph, g: ig.Graph) -> None:
    """Rewrites the name of all unknown nodes in `atk` based on their
    equivalents by UUID in `g`. Note that this change is made in-place in
    `atk`."""
    def set_name(u, name: str):
        log.info("[%s] -> [%s]", u["name"], name)
        u["name"] = name

    for u in tqdm.tqdm(atk.vs):
        if not u["name"].startswith(parse.graph.unknown_prefix):
            # skip non-unknowns
            continue
        # check without time first: some times in the new graph are slightly
        # different because they use the real timestamp and not last time
        vs = g.vs(uuid_eq=u["uuid"])
        if len(vs) == 1:
            set_name(u, vs[0]["name"])
            continue
        elif not vs:
            log.warning("[%s]: UUID (%s) not found...", u["name"], u["uuid"])
            continue
        # first, try to pinpoint via time...
        vs_time = vs(time_eq=u["time"])
        if len(vs_time) == 1:
            set_name(u, vs_time[0]["name"])
            continue
        elif len(vs_time) == 2:
            vs = vs_time
        # if len(vs_time) == 0, then use original vs

        if len(vs) == 2:
            # check if there is an exec between the two nodes
            execs = g.es(_within=(vs[0].index, vs[1].index), type_eq="exec")
            if not execs:
                log.warning("found duplicate UUIDs that aren't exec!")
                # needs manual intervention
                breakpoint()
                pass # nop for breakpoint
            elif len(execs) > 1:
                log.warning("unexpected number of execs!")
                # needs manual intervention
                breakpoint()
                pass # nop for breakpoint
            else:
                # find the duplicate in atk
                atk_us = atk.vs(uuid_eq=u["uuid"])
                if len(atk_us) == 1:
                    # one node only, this must be the child
                    log.warning("exec with one node, assuming child")
                    set_name(u, g.vs[execs[0].target]["name"])
                else:
                    # relabel both exec nodes
                    log.warning("exec with two nodes, relabeling both")
                    atk_execs = atk.es(_within=(atk_us[0].index, atk_us[1].index),
                                       type_eq="exec")
                    if len(atk_execs) == 1:
                        set_name(atk.vs[atk_execs[0].source], g.vs[execs[0].source]["name"])
                        set_name(atk.vs[atk_execs[0].target], g.vs[execs[0].target]["name"])
                    else:
                        log.warning("unexpected number of atk execs...")
                        breakpoint()
                        pass
        else:
            log.warning("found >2 duplicate UUIDs!")
            # needs manual intervention
            breakpoint()
            pass # nop for breakpoint

def relabel_e3(set_name: str) -> None:
    """Relabel all original_attack_subgraphs of `set_name` and put their
    corresponding graphs into attack_subgraphs."""
    graph_path = f"{config.e3.graphs}/{set_name}_prov_graph_relabeled_unknowns.pkl.xz"
    log.info("reading graph from %s...", graph_path)
    g = util.pickle_read(graph_path)

    src = f"{config.e3.intermediates}/original_attack_subgraphs"
    paths = glob.glob(f"{src}/{set_name}_*.pkl.xz")
    log.info("relabeling the following attack subgraphs:")
    log.info("%s", paths)
    for path in paths:
        filename = path.rsplit("/", maxsplit=1)[1]
        out_path = f"{config.e3.attack_subgraphs}/{filename}"

        #if util.skip_overwrite(out_path):
        #    log.warning("%s already exists; skipping...", out_path)
        #    continue

        log.info("reading attack subgraph from %s", path)
        atk = util.pickle_read(path)

        log.info("relabeling attack subgraph...")
        relabel_attack(atk, g)

        log.info("writing attack subgraph to %s", out_path)
        util.pickle_write(atk, out_path)

def main(argv):
    #for set_name in ["cadets", "theia", "fivedirections", "trace"]:
    #for set_name in ["cadets", "theia", "fivedirections"]:
    for set_name in ["trace"]:
        relabel_e3(set_name)
