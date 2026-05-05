"""
Per-edge structural features for downstream ML (logistic regression / GBM
on which (u,v) pairs end up in optimal attack edge sets).

The feature vector is computed for an *unordered* pair (u,v), works whether
the edge currently exists or not. Designed to be cheap so we can score all
O(n^2) candidate pairs.
"""

import math
import networkx as nx
from mu_triad import count_mu_triads_through


def edge_features(G, u, v, pred_partition, mu=10):
    """
    Args:
        G              : NetworkX graph
        u, v           : endpoints (need not currently be connected)
        pred_partition : {node: community_id} from current detector
        mu             : mu-triad threshold (only used for mu-triad features)

    Returns dict of named features.
    """
    deg = dict(G.degree())
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    common = nu & nv
    union  = nu | nv

    return {
        # degree features
        'deg_u': deg[u],
        'deg_v': deg[v],
        'deg_min': min(deg[u], deg[v]),
        'deg_max': max(deg[u], deg[v]),

        # neighborhood overlap (link prediction classics)
        'common_neighbors': len(common),
        'jaccard': len(common) / max(len(union), 1),
        'adamic_adar': sum(1.0 / math.log(deg[w]) for w in common if deg[w] > 1),

        # community-structure features
        'same_pred_comm': int(pred_partition[u] == pred_partition[v]),
        'boundary_u': len({pred_partition[w] for w in nu}),
        'boundary_v': len({pred_partition[w] for w in nv}),

        # local triangle structure
        'cc_u': nx.clustering(G, u),
        'cc_v': nx.clustering(G, v),
        'tri_u': nx.triangles(G, u),
        'tri_v': nx.triangles(G, v),

        # mu-triad participation -- the quantity mu-MEA tries to destroy
        'mu_triad_part_u': count_mu_triads_through(G, u, mu),
        'mu_triad_part_v': count_mu_triads_through(G, v, mu),
    }


def features_for_edge_set(G, edges, pred_partition, mu=10):
    """Convenience: vectorize edge_features over a list of (u,v) pairs."""
    return [edge_features(G, u, v, pred_partition, mu=mu) for (u, v) in edges]