"""Timeslice provenance graphs for use in REAPr labeling or rough estimates of
attack periods."""

import logging
import datetime
from typing import Callable, Optional, Tuple

import igraph as ig

import config
import util

log = logging.getLogger(__name__)

def _time_range(y1, m1, d1, h1, mn1,
                y2, m2, d2, h2, mn2,
                tzinfo=util.EDT) -> Tuple[int, int]:
    """Create a pair of UNIX timestamps corresponding to two dates, with the
    given year, month, day, hours, and minutes."""
    def to_ns(y, m, d, h, mn) -> int:
        dt = datetime.datetime(y, m, d, h, mn, tzinfo=tzinfo)
        return int(dt.timestamp()) * int(1e9)
    t1 = to_ns(y1, m1, d1, h1, mn1)
    t2 = to_ns(y2, m2, d2, h2, mn2)
    assert t2 > t1
    return (t1, t2)

# Assume the timestamps are EDT (Eastern Daylight Time). This is almost
# certainly correct, as the times given in `operational_event_log.md` are
# labeled "ET" and correlate with the times listed in the ground truth
# document.
#
# These ranges were collected from the ground truth, with a ~5 minute buffer
# on either end to account for possible logging artifacts, delayed behaviors,
# or approximate timestamps. For the phishing attacks, we excluded sending the
# phishing email, as this frequently occured hours before the actual
# observable attack behavior on the host itself.
e3_slices = {
    "cadets": [
        _time_range(2018, 4,  6, 11, 15,
                    2018, 4,  6, 12, 15),
        _time_range(2018, 4, 11, 15,  5,
                    2018, 4, 11, 15, 20),
        _time_range(2018, 4, 12, 13, 55,
                    2018, 4, 12, 14, 40),
        _time_range(2018, 4, 13,  9,  0,
                    2018, 4, 13,  9, 20),
    ],
    "clearscope": [
        _time_range(2018, 4,  6, 15,  0,
                    2018, 4,  6, 15, 20),
        _time_range(2018, 4, 11, 13, 50,
                    2018, 4, 11, 14, 50),
        _time_range(2018, 4, 12, 15, 15,
                    2018, 4, 12, 15, 30),
        _time_range(2018, 4, 13, 10, 25,
                    2018, 4, 13, 14, 20),
    ],
    "fivedirections": [
        # ground truth has both "TA5.2" and "FiveDirections"; we include both
        _time_range(2018, 4,  9, 13, 35,
                    2018, 4,  9, 15, 45),
        _time_range(2018, 4, 10, 13, 55,
                    2018, 4, 10, 14,  5),
        _time_range(2018, 4, 11,  9, 55,
                    2018, 4, 11, 11,  0),
        _time_range(2018, 4, 12, 10,  5,
                    2018, 4, 12, 10, 50),
        _time_range(2018, 4, 12, 11, 10,
                    2018, 4, 12, 11, 20),
    ],
    "theia": [
        _time_range(2018, 4, 10, 13, 35,
                    2018, 4, 10, 14, 55),
        _time_range(2018, 4, 12, 12, 40,
                    2018, 4, 12, 13, 30),
        _time_range(2018, 4, 13, 13, 45,
                    2018, 4, 13, 14, 10),
    ],
    "trace": [
        _time_range(2018, 4, 10,  9, 40,
                    2018, 4, 10, 11, 15),
        _time_range(2018, 4, 10, 12, 25,
                    2018, 4, 10, 12, 35),
        _time_range(2018, 4, 12, 13, 30,
                    2018, 4, 12, 13, 40),
        _time_range(2018, 4, 13, 12, 40,
                    2018, 4, 13, 12, 55),
    ],
}

# All EDT. These were collected via the ground truth document, with a ~5
# minute buffer on both ends.
optc_slices = {
    "host201": [
        _time_range(2019, 9, 23, 11, 20,
                    2019, 9, 23, 15, 35),
    ],
    "host402": [
        _time_range(2019, 9, 23, 13, 20,
                    2019, 9, 23, 14, 10),
    ],
    "host660": [
        _time_range(2019, 9, 23, 13, 30,
                    2019, 9, 23, 14, 10),
    ],
    "DC1 VM": [ # does not appear to be within dataset
        _time_range(2019, 9, 23, 14,  0,
                    2019, 9, 23, 15, 35),
    ],
    "host501": [
        _time_range(2019, 9, 24, 10, 20,
                    2019, 9, 25, 10,  5),
    ],
    "host811": [
        _time_range(2019, 9, 24, 10, 35,
                    2019, 9, 24, 10, 45),
    ],
    "dc1": [ # does not appear to be within dataset
        _time_range(2019, 9, 23, 11, 15,
                    2019, 9, 23, 14, 10),
    ],
    "host974": [
        _time_range(2019, 9, 24, 13, 40,
                    2019, 9, 24, 15, 35),
    ],
    "host5": [
        _time_range(2019, 9, 24, 13, 50,
                    2019, 9, 24, 15, 35),
    ],
    "host51": [ # note: September 26 is not in the released data
        _time_range(2019, 9, 26, 10, 25,
                    2019, 9, 26, 14, 30),
    ],
}

def _get_slices(cfg) -> dict:
    if cfg == config.e3:
        return e3_slices
    elif cfg == config.optc:
        return optc_slices
    else:
        raise ValueError("no defined slices for the given config!")

def shade_attack(cfg, set_name: str) -> Optional[Callable]:
    """Return a function that takes `matplotlib.pyplot` and shades the attack
    period in translucent red, or None if no attack periods were found.
    Intended to be used for neighborhood plots."""
    try:
        slices = _get_slices(cfg)
    except ValueError:
        return None

    if set_name not in slices:
        return None

    def _shade(plt: "matplotlib.pyplot") -> None:
        for s in slices[set_name]:
            start, end = map(lambda t: t / 1e9, s) # convert back to seconds
            plt.axvspan(start, end, facecolor="#ff0000", alpha=0.7, zorder=10)

    return _shade

def slice_graph(cfg, set_name: str, g: ig.Graph,
                slice_name: str = None):
    """Slice a graph based on its ground truth attack periods. Can optionally
    specify the slice name as above if the set name is different.

    If multiple slices exist, this generates a separate graph per slice."""
    if slice_name is None:
        slice_name = set_name
    slices = _get_slices(cfg)

    vcount = g.vcount()
    ecount = g.ecount()
    vpad = len(str(vcount))
    epad = len(str(ecount))
    log.info("Graph: |V| = %s, |E| = %s", vcount, ecount)

    for i, (start, end) in enumerate(slices[slice_name]):
        # remove edges outside timeslice
        g.delete_edges(time_le=start)
        g.delete_edges(time_ge=end)

        # delete any isolated vertices
        g.delete_vertices(g.vs(_degree_eq=0))

        log.info("Slice: |V| = %*s, |E| = %*s",
                 vpad, g.vcount(), epad, g.ecount())

        # write output to intermediates
        out_path = f"{cfg.timesliced_graphs}/{set_name}_slice{i}.pkl.xz"
        log.info("writing %s...", out_path)
        util.pickle_write(g, out_path)

def write_optc_slices():
    graphs = [201, 402, 660, 501, 811, 974, 5, 51]
    for n in graphs:
        set_name = f"host{n}"
        graph_path = f"{config.optc.graphs}/{set_name}_prov_graph.pkl.xz"
        log.info("reading %s...", graph_path)
        g = util.pickle_read(graph_path)
        slice_graph(config.optc, set_name, g)

def main(argv):
    write_optc_slices()
