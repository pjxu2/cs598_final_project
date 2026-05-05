"""
Thin wrappers around igraph + leidenalg so the rest of the codebase can stay
in NetworkX. Two community detection algorithms are exposed:

  - louvain_q : Louvain-style modularity optimization (via Leiden's
                ModularityVertexPartition).
  - leiden_cpm: Leiden with the Constant Potts Model objective at a given
                resolution gamma. Smaller gamma => larger communities.

Both return a dict {node_id: community_id}, keyed by integer node IDs that
match the NetworkX graph's nodes (which we relabel to 0..n-1 in graphs.py).
"""

import igraph as ig
import leidenalg as la


def nx_to_ig(G):
    """Convert an undirected NetworkX graph (with integer 0..n-1 labels) to igraph."""
    return ig.Graph(n=G.number_of_nodes(),
                    edges=list(G.edges()),
                    directed=False)


def louvain_q(G):
    """Modularity-optimizing partition. Returns {node: community_id}."""
    g = nx_to_ig(G)
    part = la.find_partition(g, la.ModularityVertexPartition, seed=0)
    return {v: part.membership[v] for v in range(G.number_of_nodes())}


def leiden_cpm(G, gamma=0.05):
    """CPM partition with resolution gamma. Returns {node: community_id}."""
    g = nx_to_ig(G)
    part = la.find_partition(g, la.CPMVertexPartition,
                             resolution_parameter=gamma, seed=0)
    return {v: part.membership[v] for v in range(G.number_of_nodes())}