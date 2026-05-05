"""
Main experimental comparison: muMEA vs SA-based attacks across graph
categories, with MATCHED k between attackers.

Per (instance, budget, k):
  - muMEA-best-at-k : max ARI/NMI reduction over mu_thresh, fixed k.
  - SA(Q-ARI)       : SA targeting Louvain-Q ARI, k_per_vertex=k.
  - SA(CPM-ARI)     : SA targeting Leiden-CPM ARI, k_per_vertex=k.

We also save each attack's chosen edge set to results/attack_edges.json
(keyed by attack_id) so we can later study which edges make a "good" attack.

Outputs:
  results/experiment_results.csv : one row per (instance, budget, k, attacker)
  results/attack_edges.json      : {attack_id: [[u,v], ...]}
"""

import csv
import json
import os
import statistics
import time
from collections import defaultdict

from cd_algos import leiden_cpm, louvain_q
from datasets_loader import (gen_lfr, iter_ec_sbm, load_cdlib_real,
                             load_karate)
from metrics import eval_attack
from mu_triad import mu_mea
from objectives import make_obj_ari_drop
from sa_attack import sa_attack


# ----------------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------------


BUDGET_PCTS      = [0.05, 0.10]

# single k value to halve SA workload; can expand later if time permits
K_VALUES         = [5, 10]

MUMEA_MU_THRESHS = [10, 15, 20]

SA_ITERS         = 1500        # was 1500
GAMMA_CPM        = 0.05

# only LFR n=200 (3 seeds). 400 doubles SA time.
LFR_SIZES = [200]
LFR_SEEDS = [0, 1]
LFR_MU    = 0.3

EC_SBM_ROOT      = "datasets"
EC_SBM_LIMIT     = 4
EC_SBM_MAX_NODES = 2000        # NEW: skip instances bigger than this

INCLUDE_KARATE = True
CDLIB_NETS     = []            # disabled

OUTPUT_CSV   = "results/experiment_results.csv"
OUTPUT_EDGES = "results/attack_edges.json"


# ----------------------------------------------------------------------
#  Per-attack helpers
# ----------------------------------------------------------------------

def _reductions(G_orig, G_att, gt):
    """Δ-ARI / Δ-NMI for both detectors."""
    q  = eval_attack(G_orig, G_att, gt, louvain_q)
    cp = eval_attack(G_orig, G_att, gt, lambda H: leiden_cpm(H, GAMMA_CPM))
    return {
        'd_ari_q':   q['ari_before']  - q['ari_after'],
        'd_nmi_q':   q['nmi_before']  - q['nmi_after'],
        'd_ari_cpm': cp['ari_before'] - cp['ari_after'],
        'd_nmi_cpm': cp['nmi_before'] - cp['nmi_after'],
    }


def _edges_added(G_orig, G_att):
    """Edges in G_att but not G_orig, normalized to (min,max) tuples."""
    orig = {(min(u, v), max(u, v)) for u, v in G_orig.edges()}
    att  = {(min(u, v), max(u, v)) for u, v in G_att.edges()}
    return list(att - orig)


def run_mumea_at_k(G, gt, budget, k):
    """
    Sweep muMEA over mu_thresh for fixed k. Returns:
      best_red       : max-over-mu reduction for each metric
      best_mu        : {metric: mu_thresh that achieved it}
      best_edges     : {metric: list of (u,v) for the winning config}
      per_mu_rows    : raw per-config records
    """
    best_red = {'d_ari_q': float('-inf'), 'd_nmi_q': float('-inf'),
                'd_ari_cpm': float('-inf'), 'd_nmi_cpm': float('-inf')}
    best_mu = {}
    best_edges = {}
    per_mu_rows = []

    for mu in MUMEA_MU_THRESHS:
        try:
            G_att = mu_mea(G, b=budget, k=k, mu=mu)
        except Exception:
            continue
        red = _reductions(G, G_att, gt)
        edges = _edges_added(G, G_att)
        per_mu_rows.append({'k': k, 'mu_thresh': mu,
                            'edges': edges, **red})
        for metric, v in red.items():
            if v > best_red[metric]:
                best_red[metric] = v
                best_mu[metric] = mu
                best_edges[metric] = edges

    for m in best_red:
        if best_red[m] == float('-inf'):
            best_red[m] = 0.0
    return best_red, best_mu, best_edges, per_mu_rows


def run_sa(G, gt, budget, detect_fn, k_per_vertex, n_iter=SA_ITERS):
    """SA targeting detect_fn ARI with k-unnoticeability. Returns red + edges."""
    obj = make_obj_ari_drop(gt, detect_fn)
    added, _, _, _ = sa_attack(G, budget, obj,
                               k_per_vertex=k_per_vertex,
                               n_iter=n_iter, verbose=False)
    G_att = G.copy(); G_att.add_edges_from(added)
    edges = [(min(u, v), max(u, v)) for u, v in added]
    return _reductions(G, G_att, gt), edges


# ----------------------------------------------------------------------
#  Instance gathering
# ----------------------------------------------------------------------
"""
def gather_instances():
    # Yield (category, name, G, gt).
    # LFR
    for n in LFR_SIZES:
        for seed in LFR_SEEDS:
            try:
                G, gt = gen_lfr(n=n, mu=LFR_MU, seed=seed)
                yield ("LFR", f"n{n}_s{seed}", G, gt)
            except RuntimeError as e:
                print(f"  [skip LFR n={n} seed={seed}: {e}]")

    # EC-SBM (single root, walked recursively)
    if EC_SBM_ROOT and os.path.isdir(EC_SBM_ROOT):
        cnt = 0
        for inst_name, G, gt in iter_ec_sbm(EC_SBM_ROOT):
            yield ("EC-SBM", inst_name.replace(os.sep, "_"), G, gt)
            cnt += 1
            if cnt >= EC_SBM_LIMIT:
                break

    # Real-world
    if INCLUDE_KARATE:
        G, gt = load_karate()
        yield ("Real", "karate", G, gt)

    for net in CDLIB_NETS:
        try:
            G, gt = load_cdlib_real(net, max_nodes=CDLIB_MAX_NODES)
            yield ("Real", net, G, gt)
        except Exception as e:
            print(f"  [skip cdlib '{net}': {e}]")

"""

def gather_instances():
    # LFR
    for n in LFR_SIZES:
        for seed in LFR_SEEDS:
            try:
                G, gt = gen_lfr(n=n, mu=LFR_MU, seed=seed)
                yield ("LFR", f"n{n}_s{seed}", G, gt)
            except RuntimeError as e:
                print(f"  [skip LFR n={n} seed={seed}: {e}]")

    # EC-SBM with size filter
    if EC_SBM_ROOT and os.path.isdir(EC_SBM_ROOT):
        cnt = 0
        for inst_name, G, gt in iter_ec_sbm(EC_SBM_ROOT):
            n = G.number_of_nodes()
            if n > EC_SBM_MAX_NODES:
                print(f"  [skip EC-SBM/{inst_name}: {n} > {EC_SBM_MAX_NODES} nodes]")
                continue
            # store path with forward-slash so we can reload later
            stored = inst_name.replace(os.sep, "/")
            yield ("EC-SBM", stored, G, gt)
            cnt += 1
            if cnt >= EC_SBM_LIMIT:
                break

    if INCLUDE_KARATE:
        G, gt = load_karate()
        yield ("Real", "karate", G, gt)

    for net in CDLIB_NETS:
        try:
            G, gt = load_cdlib_real(net, max_nodes=2500,
                                    min_csize=10, max_csize=200)
            yield ("Real", net, G, gt)
        except Exception as e:
            print(f"  [skip cdlib '{net}': {e}]")



# ----------------------------------------------------------------------
#  Main loop
# ----------------------------------------------------------------------

def _attack_id(category, instance, budget, attacker, k, mu_thresh=None):
    base = f"{category}__{instance}__b{budget}__{attacker}__k{k}"
    if mu_thresh is not None:
        base += f"__mu{mu_thresh}"
    return base

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    fieldnames = ['attack_id', 'category', 'instance', 'n', 'm',
                  'budget_pct', 'budget', 'attacker', 'k', 'mu_thresh',
                  'd_ari_q', 'd_nmi_q', 'd_ari_cpm', 'd_nmi_cpm']

    # open CSV up front and flush after each instance
    csv_fh = open(OUTPUT_CSV, 'w', newline='')
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()
    csv_fh.flush()

    edge_dump = {}
    rows_total = 0

    instances = list(gather_instances())
    print(f"Total instances: {len(instances)}\n")

    t_start = time.time()
    for category, name, G, gt in instances:
        n, m = G.number_of_nodes(), G.number_of_edges()
        local_rows = []

        for bp in BUDGET_PCTS:
            budget = int(bp * m)
            if budget < 1:
                continue
            print(f"[{category}/{name}] |V|={n} |E|={m} "
                  f"budget={budget} ({bp*100:.1f}%)")

            for k in K_VALUES:
                t0 = time.time()
                red_mu, best_mu, best_edges, per_mu_rows = \
                    run_mumea_at_k(G, gt, budget, k)
                t_mu = time.time() - t0

                for r in per_mu_rows:
                    aid = _attack_id(category, name, budget,
                                     "muMEA", k=k, mu_thresh=r['mu_thresh'])
                    edge_dump[aid] = r['edges']
                    local_rows.append({
                        'attack_id': aid, 'category': category,
                        'instance': name, 'n': n, 'm': m,
                        'budget_pct': bp, 'budget': budget,
                        'attacker': 'muMEA', 'k': k,
                        'mu_thresh': r['mu_thresh'],
                        'd_ari_q': r['d_ari_q'], 'd_nmi_q': r['d_nmi_q'],
                        'd_ari_cpm': r['d_ari_cpm'], 'd_nmi_cpm': r['d_nmi_cpm'],
                    })

                local_rows.append({
                    'attack_id': _attack_id(category, name, budget,
                                            "muMEA-best", k=k),
                    'category': category, 'instance': name, 'n': n, 'm': m,
                    'budget_pct': bp, 'budget': budget,
                    'attacker': 'muMEA-best', 'k': k, 'mu_thresh': None,
                    **red_mu,
                })

                t0 = time.time()
                red_q, edges_q = run_sa(G, gt, budget, louvain_q,
                                        k_per_vertex=k)
                t_q = time.time() - t0
                aid = _attack_id(category, name, budget, "SA-Q", k=k)
                edge_dump[aid] = edges_q
                local_rows.append({
                    'attack_id': aid, 'category': category,
                    'instance': name, 'n': n, 'm': m,
                    'budget_pct': bp, 'budget': budget,
                    'attacker': 'SA-Q', 'k': k, 'mu_thresh': None,
                    **red_q,
                })

                t0 = time.time()
                red_cpm, edges_cpm = run_sa(
                    G, gt, budget,
                    lambda H: leiden_cpm(H, GAMMA_CPM),
                    k_per_vertex=k)
                t_cpm = time.time() - t0
                aid = _attack_id(category, name, budget, "SA-CPM", k=k)
                edge_dump[aid] = edges_cpm
                local_rows.append({
                    'attack_id': aid, 'category': category,
                    'instance': name, 'n': n, 'm': m,
                    'budget_pct': bp, 'budget': budget,
                    'attacker': 'SA-CPM', 'k': k, 'mu_thresh': None,
                    **red_cpm,
                })

                print(f"   k={k}: muMEA-best ΔARI_Q={red_mu['d_ari_q']:+.3f} "
                      f"ΔARI_CPM={red_mu['d_ari_cpm']:+.3f} ({t_mu:.1f}s) | "
                      f"SA-Q ΔQ={red_q['d_ari_q']:+.3f} ({t_q:.1f}s) | "
                      f"SA-CPM ΔCPM={red_cpm['d_ari_cpm']:+.3f} ({t_cpm:.1f}s)")

        # flush this instance's rows to disk + dump edges so far
        writer.writerows(local_rows)
        csv_fh.flush()
        rows_total += len(local_rows)
        with open(OUTPUT_EDGES, 'w') as fh:
            json.dump(edge_dump, fh)
        print(f"  [saved {rows_total} rows so far]")

    csv_fh.close()
    print(f"\nTotal wallclock: {(time.time()-t_start)/60:.1f} min")
    print(f"Wrote {rows_total} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(edge_dump)} edge sets to {OUTPUT_EDGES}")

    # summary table (read CSV back so we can compute even on partial data)
    headline = {'muMEA-best', 'SA-Q', 'SA-CPM'}
    grouped = defaultdict(list)
    with open(OUTPUT_CSV, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['attacker'] in headline:
                grouped[(r['category'], r['budget_pct'],
                         r['k'], r['attacker'])].append(r)

    print("\n========= MEAN REDUCTIONS =========")
    print(f"{'cat':<8} {'bp':>5} {'k':>3} {'attacker':<12} "
          f"{'ΔARI_Q':>8} {'ΔARI_CPM':>10} "
          f"{'ΔNMI_Q':>8} {'ΔNMI_CPM':>10}  {'#':>3}")
    for (cat, bp, k, atk), recs in sorted(grouped.items()):
        mean = lambda key: statistics.mean(float(r[key]) for r in recs)
        print(f"{cat:<8} {bp:>5} {k:>3} {atk:<12} "
              f"{mean('d_ari_q'):>+8.3f} {mean('d_ari_cpm'):>+10.3f} "
              f"{mean('d_nmi_q'):>+8.3f} {mean('d_nmi_cpm'):>+10.3f}  "
              f"{len(recs):>3d}")

if __name__ == "__main__":
    main()