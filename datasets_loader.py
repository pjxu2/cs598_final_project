"""
Unified dataset loader. Every function returns (G, gt) where:
  - G  : nx.Graph, nodes relabeled to 0..n-1
  - gt : {node_id: community_id} ground truth labels (or None if unavailable)

Synthetic:
  - gen_lfr : LFR benchmark (configurable mu)
  - gen_sbm : Stochastic block model with planted equally-sized communities

Real-world (ground truth available):
  - load_karate   : Zachary's karate club (34 nodes, 2 communities)
  - load_football : American football games (115 nodes, ~12 conferences) [needs file]
  - load_email_eu : email-Eu-core (1005 nodes, 42 depts) [needs file]

EC-SBM (file-based, expects {instance}/edge.tsv + {instance}/com.tsv):
  - load_ec_sbm  : load a single EC-SBM instance directory
  - iter_ec_sbm  : iterate over all instances in a parent directory

If you don't have ground truth (e.g. arbitrary real-world graph), pass
gt=None downstream and stick to objectives that don't need it
(e.g. modularity drop).
"""

import os
import networkx as nx
import numpy as np
from networkx.generators.community import LFR_benchmark_graph


# ---------- internal utilities ----------

def _relabel_and_cleanup(G):
    """Make G simple, undirected, integer-labeled 0..n-1."""
    G = nx.Graph(G)                                 # drop multiedges/direction
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
    return G


# ---------- synthetic ----------

def gen_lfr(n=300, tau1=3, tau2=1.5, mu=0.3, avg_deg=10,
            min_comm=30, max_comm=80, max_deg=None, seed=0, max_tries=20):
    """LFR benchmark. Retries seeds because LFR generation can fail."""
    if max_deg is None:
        max_deg = n // 4

    last_err = None
    for s in range(seed, seed + max_tries):
        try:
            G = LFR_benchmark_graph(
                n, tau1, tau2, mu,
                average_degree=avg_deg,
                min_community=min_comm, max_community=max_comm,
                max_degree=max_deg, seed=s,
            )
            comm_attr = {v: G.nodes[v]['community'] for v in G.nodes}
            G = _relabel_and_cleanup(G)
            # remap ground truth via 'orig_id'
            gt = {v: hash(frozenset(comm_attr[G.nodes[v]['orig_id']]))
                  for v in G.nodes}
            return G, gt
        except (nx.ExceededMaxIterations, nx.PowerIterationFailedConvergence) as e:
            last_err = e
            continue
    raise RuntimeError(f"LFR failed {max_tries} times: {last_err}")


def gen_sbm(sizes=(75, 75, 75, 75), p_in=0.15, p_out=0.01, seed=0):
    """
    Stochastic block model with planted communities.
    sizes: tuple of community sizes (sum = n).
    p_in : within-community edge probability
    p_out: between-community edge probability
    """
    sizes = list(sizes)
    p = [[p_in if i == j else p_out for j in range(len(sizes))]
         for i in range(len(sizes))]
    G = nx.stochastic_block_model(sizes, p, seed=seed)
    # SBM gives node attribute 'block' as ground-truth community id
    gt_raw = {v: G.nodes[v]['block'] for v in G.nodes}
    G = _relabel_and_cleanup(G)
    gt = {v: gt_raw[G.nodes[v]['orig_id']] for v in G.nodes}
    return G, gt


# ---------- real-world ----------

def load_karate():
    """Zachary's karate club. Communities = 'club' attribute (Mr Hi vs Officer)."""
    G = nx.karate_club_graph()
    raw = {v: G.nodes[v]['club'] for v in G.nodes}
    G = _relabel_and_cleanup(G)
    label_map = {lbl: i for i, lbl in enumerate(set(raw.values()))}
    gt = {v: label_map[raw[G.nodes[v]['orig_id']]] for v in G.nodes}
    return G, gt


def load_edgelist_with_communities(edges_path, comm_path, sep=None):
    """
    Generic loader. Useful for SNAP-style datasets like email-Eu-core.
      edges_path: file with one 'u v' per line
      comm_path : file with one 'node community' per line
    Both spaces and tabs allowed (sep=None splits on whitespace).
    """
    G = nx.read_edgelist(edges_path, nodetype=int)
    raw = {}
    with open(comm_path) as fh:
        for line in fh:
            parts = line.strip().split(sep)
            if len(parts) >= 2:
                raw[int(parts[0])] = int(parts[1])
    G = _relabel_and_cleanup(G)
    gt = {v: raw[G.nodes[v]['orig_id']] for v in G.nodes
          if G.nodes[v]['orig_id'] in raw}
    return G, gt


def load_football(gml_path):
    """
    American college football network (Girvan & Newman 2002).
    Download football.gml from http://www-personal.umich.edu/~mejn/netdata/
    Community = 'value' attribute (conference index).
    """
    G = nx.read_gml(gml_path)
    raw = {v: G.nodes[v]['value'] for v in G.nodes}
    G = _relabel_and_cleanup(G)
    gt = {v: raw[G.nodes[v]['orig_id']] for v in G.nodes}
    return G, gt


# ---------- EC-SBM ----------

def load_ec_sbm(instance_dir):
    """
    Load a single EC-SBM instance from a directory containing:
      - edge.tsv : tab-separated edge list, one 'u\tv' per line
      - com.tsv  : tab-separated 'node_id\tcommunity_id' per line

    Example:
        G, gt = load_ec_sbm("datasets/internet_as")

    Skips a header row if the first line isn't all-numeric. Both files are
    expected to use the same node IDs (we relabel both consistently to 0..n-1
    at the end).
    """
    edge_path = os.path.join(instance_dir, "edge.tsv")
    com_path  = os.path.join(instance_dir, "com.tsv")
    if not (os.path.isfile(edge_path) and os.path.isfile(com_path)):
        raise FileNotFoundError(
            f"Expected edge.tsv and com.tsv inside {instance_dir}")

    # --- edges ---
    G = nx.Graph()
    with open(edge_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue  # header or malformed row
            if u != v:
                G.add_edge(u, v)

    # --- communities ---
    raw = {}
    with open(com_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                node, comm = int(parts[0]), int(parts[1])
            except ValueError:
                continue  # header
            raw[node] = comm

    # Make sure every node in G has a label (some EC-SBM dumps have isolates
    # only listed in com.tsv; add them so gt is total).
    labeled_nodes = set(raw.keys()) & set(G.nodes())
    G = G.subgraph(labeled_nodes).copy()

    # If that left the graph disconnected, take the largest connected
    # component for clean attack semantics.
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        ccs = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(ccs[0]).copy()

    G = _relabel_and_cleanup(G)
    gt = {v: raw[G.nodes[v]['orig_id']] for v in G.nodes}
    return G, gt


def iter_ec_sbm(root_dir):
    """
    Yield (instance_name, G, gt) for every subdirectory of `root_dir` that
    contains edge.tsv + com.tsv. Useful for sweeping across instances.

    Example:
        for name, G, gt in iter_ec_sbm("datasets/internet_as"):
            print(name, G.number_of_nodes(), G.number_of_edges())
    """
    for entry in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, entry)
        if not os.path.isdir(sub):
            continue
        if os.path.isfile(os.path.join(sub, "edge.tsv")) \
                and os.path.isfile(os.path.join(sub, "com.tsv")):
            G, gt = load_ec_sbm(sub)
            yield entry, G, gt



def load_cdlib_real(name, max_nodes=2500, min_csize=20, max_csize=400, seed=0):
    """
    Load a real-world network with overlapping ground truth via cdlib, then
    subsample down to a tractable size for SA.

    Strategy:
      1. Fetch full graph + community list from cdlib.
      2. Pick communities of moderate size (in [min_csize, max_csize]),
         largest first, until we cover ~max_nodes.
      3. Induce subgraph on the union of those community nodes.
      4. Take the largest connected component.
      5. Make ground truth non-overlapping by assigning each node to its
         first listed community membership.

    Common names: 'amazon', 'youtube', 'dblp', 'karate'.
    Requires `pip install cdlib`.
    """
    try:
        from cdlib.datasets import fetch_network_data, fetch_ground_truth_data
    except ImportError as e:
        raise ImportError("Install cdlib first: pip install cdlib") from e

    G_raw = fetch_network_data(net_name=name, net_type="networkx")
    nc    = fetch_ground_truth_data(net_name=name)
    coms_list = nc.communities  # list[list[node]]; possibly overlapping

    # filter to moderate-size communities and sort largest-first
    eligible = [c for c in coms_list if min_csize <= len(c) <= max_csize]
    eligible.sort(key=len, reverse=True)

    chosen, node_set = [], set()
    for c in eligible:
        if len(node_set) >= max_nodes:
            break
        new = node_set | set(c)
        if len(new) <= int(max_nodes * 1.2):
            chosen.append(c)
            node_set = new

    if not chosen:
        # fallback: just take the largest few communities regardless of size
        for c in sorted(coms_list, key=len, reverse=True)[:5]:
            node_set |= set(c)
            chosen.append(c)

    G_sub = G_raw.subgraph(node_set).copy()

    # largest connected component (the induced subgraph is rarely connected)
    if not nx.is_connected(G_sub):
        ccs = sorted(nx.connected_components(G_sub), key=len, reverse=True)
        G_sub = G_sub.subgraph(ccs[0]).copy()

    # non-overlapping labels: first community a node appears in wins
    raw = {}
    for cid, members in enumerate(chosen):
        for u in members:
            if u in G_sub and u not in raw:
                raw[u] = cid

    # drop nodes without a label (shouldn't happen often)
    G_sub = G_sub.subgraph(list(raw.keys())).copy()
    G = _relabel_and_cleanup(G_sub)
    gt = {v: raw[G.nodes[v]['orig_id']] for v in G.nodes}
    return G, gt