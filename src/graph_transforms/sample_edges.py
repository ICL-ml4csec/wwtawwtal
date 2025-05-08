"""Sample edges for edge-based systems. Needed for FLASH, which does edge
embeddings, so we cannot simply reduce down to 1 edge. Maybe is also useful
for Kairos if that has space problems.

The problematic graph seems to be Trace, where one of the failed attacks has
extreme (>1 million) communication between two nodes. The approach here is to
check if # edges >>> # neighbors, and if so, sample by only retaining a random
selection of edges."""

import random
from typing import List

import igraph as ig
import tqdm

import config
import util

def sample_edges(g: ig.Graph, threshold: int = int(1e4)):
    """Sample edges of a graph in-place: for any node whose degree exceeds
    `threshold`, reduce its degree to approximately `threshold`. This will
    only ever remove parallel edges, and never change the connectivity of the
    graph."""
    rng = random.Random(42) # use fixed seed for deterministic results
    remove = []
    for u in tqdm.tqdm(g.vs, total=g.vcount()):
        deg = u.degree(mode="all", loops=True)
        if deg > threshold:
            # strategy: tally up counts and keep random edges
            # keep at least 1 kind of every edge, hence "approximate": if the
            # proportion should result in 0 edges, this will be slightly over
            i_keep = set(rng.sample(range(deg), k=threshold))
            nbrs_in = {}
            nbrs_out = {}
            for i, e in tqdm.tqdm(enumerate(u.incident(mode="all")),
                                  desc="Sampling edges", total=deg):
                # find directionality and other node (edge case is any edge
                # (u, u), but those shouldn't exist in prov graphs)
                if e.source == u.index:
                    nbrs = nbrs_out
                    v = e.target
                else:
                    nbrs = nbrs_in
                    v = e.source
                # avoid double-counting: if other is smaller, skip
                # this guarantees each edge is only ever examined once
                if v < u.index:
                    continue

                keep = i in i_keep
                if v not in nbrs:
                    nbrs[v] = set()
                if e["type"] not in nbrs[v]:
                    nbrs[v].add(e["type"])
                    # always add first type of each edge, even if not sampled
                    keep = True
                if not keep:
                    remove.append(e.index)

    g.delete_edges(remove)
