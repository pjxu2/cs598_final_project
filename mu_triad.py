"""
mu-MEA wrapper plus a few mu-triad combinatorial helpers.

A *mu-triad* (Hsu et al.) is a triangle where every vertex has degree <= mu.
The intuition: low-degree triangles are 'community evidence'; killing them
hurts modularity-based clustering. mu-MEA is the approximation algorithm
they propose to greedily flip edges that destroy many mu-triads.

We import their reference implementation (TRIAD) from a local clone of the
authors' repo. The path below points there -- adjust if you move the repo.
"""

import os
import sys

# Path to local clone of the authors' repository containing TRIAD.py.
_TRIAD_REPO = r"C:\Users\perim\Documents\College\CS598\Community Detection\cs598_final_project\hsu_triad"

if _TRIAD_REPO not in sys.path:
    sys.path.insert(0, _TRIAD_REPO)
from TRIAD import TRIAD  # noqa: E402


# ---------- combinatorial helpers ----------

def find_triangles(G):
    """Yield each triangle (u,v,w) exactly once with u<v<w."""
    for u in G.nodes:
        nbrs_u = set(G.neighbors(u))
        for v in nbrs_u:
            if v <= u:
                continue
            for w in set(G.neighbors(v)) & nbrs_u:
                if w <= v:
                    continue
                yield (u, v, w)


def count_mu_triads(G, mu):
    """Total number of triangles with every vertex of degree <= mu."""
    deg = dict(G.degree())
    return sum(1 for u, v, w in find_triangles(G)
               if deg[u] <= mu and deg[v] <= mu and deg[w] <= mu)


def count_mu_triads_through(G, v, mu):
    """Number of mu-triads that vertex v participates in."""
    deg = dict(G.degree())
    if deg[v] > mu:
        return 0
    nbrs = [w for w in G.neighbors(v) if deg[w] <= mu]
    nbrs_set = set(nbrs)
    cnt = 0
    for i, a in enumerate(nbrs):
        for b in nbrs[i+1:]:
            if b in nbrs_set and G.has_edge(a, b):
                cnt += 1
    return cnt


# ---------- mu-MEA attack ----------

class _GraphWrap:
    """TRIAD expects an object with a `.G` NetworkX attribute."""
    def __init__(self, G):
        self.G = G


def mu_mea(G, b, k=10, mu=10):
    """
    Run the authors' mu-MEA attack.

    Args:
        G   : NetworkX graph
        b   : edge budget (# edges to add)
        k   : per-vertex new-neighbor cap (their hyperparameter)
        mu  : mu-triad degree threshold

    Returns:
        G_attacked: NEW NetworkX graph = G + added edges.
    """
    attacker = TRIAD(
        graph=_GraphWrap(G),
        budget=b,
        degree=mu,          # their 'degree' arg is our mu
        k=k,
        CDA=None,           # unused by their code path
        added_edge_num=b,
    )
    result = attacker.run()

    # TRIAD.run() in the upstream code returns the modified graph; if your
    # local fork returns an edge list instead, fall back to that.
    if hasattr(result, 'edges'):
        return result
    G_att = G.copy()
    G_att.add_edges_from(result)
    return G_att