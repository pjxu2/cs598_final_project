"""
Feature analysis on the attack edges produced by experiments.py.

Pipeline:
  1. Load experiment_results.csv + attack_edges.json.
  2. For each "strong" attack (top half by Δ-ARI within attacker type):
     - Reload the original graph G + ground truth.
     - Positive examples: edges in the attack edge set.
     - Negative examples: random non-edges of equal count, matched degree.
     - Compute structural features for each pair.
  3. Stack into a design matrix and fit:
     - Logistic regression (interpretable coefficients)
     - Gradient boosting (predictive accuracy + nonlinear feature importance)
  4. Compare feature distributions across attacker types
     (SA-Q vs SA-CPM vs muMEA-best) to see if they pick qualitatively
     different edges.
"""

import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cd_algos import leiden_cpm, louvain_q
from datasets_loader import gen_lfr, load_ec_sbm, load_karate
from features import edge_features


# ----------------------------------------------------------------------
#  Config (must match experiments.py)
# ----------------------------------------------------------------------
RESULTS_CSV  = "results/experiment_results.csv"
EDGES_JSON   = "results/attack_edges.json"
EC_SBM_ROOT  = "datasets"
LFR_MU       = 0.3
GAMMA_CPM    = 0.05
MU_FOR_FEATS = 15      # mu used inside features.edge_features

RANDOM_SEED  = 0
NEG_PER_POS  = 1       # ratio of negative non-edges to positive edges
TOP_QUANTILE = 0.5     # keep top half of attacks by Δ-ARI


# ----------------------------------------------------------------------
#  Re-load helpers
# ----------------------------------------------------------------------

def reload_instance(category, instance):
    """Recreate the (G, gt) pair that experiments.py used for this row."""
    if category == "LFR":
        # "n200_s0" -> n=200 seed=0
        parts = instance.split("_")
        n    = int(parts[0][1:])
        seed = int(parts[1][1:])
        return gen_lfr(n=n, mu=LFR_MU, seed=seed)
    if category == "EC-SBM":
        path = os.path.join(EC_SBM_ROOT, *instance.split("/"))
        return load_ec_sbm(path)
    if category == "Real":
        if instance == "karate":
            return load_karate()
        raise ValueError(f"Unknown real instance: {instance}")
    raise ValueError(f"Unknown category: {category}")


def detector_for(attacker):
    """Match the SA target's detector for the predicted partition."""
    if attacker == "SA-CPM":
        return lambda H: leiden_cpm(H, GAMMA_CPM)
    return louvain_q   # SA-Q, muMEA-best, muMEA-* all default to Louvain


# ----------------------------------------------------------------------
#  Build design matrix
# ----------------------------------------------------------------------

def sample_negatives(G, positives, n_neg, rng):
    """Sample non-edges as the negative class."""
    nodes = list(G.nodes)
    pos_set = {tuple(sorted(e)) for e in positives}
    negs = set()
    tries = 0
    while len(negs) < n_neg and tries < 50 * n_neg:
        u, v = rng.sample(nodes, 2)
        e = tuple(sorted((u, v)))
        if e in pos_set or e in negs or G.has_edge(u, v):
            tries += 1
            continue
        negs.add(e)
        tries += 1
    return list(negs)


def featurize(G, pred, edges):
    """Return list of feature dicts for an iterable of (u,v) tuples."""
    return [edge_features(G, u, v, pred, mu=MU_FOR_FEATS) for (u, v) in edges]


def build_dataset():
    """
    Read CSV + JSON, produce a single (X, y, groups) DataFrame where:
      - y=1: edge was chosen by a strong attack
      - y=0: random non-edge sampled from the same graph
      - 'attacker' column lets us slice by method
    """
    df = pd.read_csv(RESULTS_CSV)
    with open(EDGES_JSON) as fh:
        edges_by_id = json.load(fh)

    # Pick "strong" attacks: top quantile of d_ari_q (or d_ari_cpm for SA-CPM)
    headline_attackers = ['SA-Q', 'SA-CPM', 'muMEA-best']
    strong_ids = []
    for atk in headline_attackers:
        sub = df[df['attacker'] == atk].copy()
        if sub.empty:
            continue
        metric = 'd_ari_cpm' if atk == 'SA-CPM' else 'd_ari_q'
        cutoff = sub[metric].quantile(TOP_QUANTILE)
        sub = sub[sub[metric] >= cutoff]
        strong_ids.extend(sub['attack_id'].tolist())
    print(f"Selected {len(strong_ids)} strong attacks")

    rng = random.Random(RANDOM_SEED)
    all_records = []

    # cache reloaded graphs; many attacks share the same instance
    graph_cache = {}

    for _, row in df.iterrows():
        if row['attack_id'] not in strong_ids:
            continue
        if row['attack_id'] not in edges_by_id:
            continue
        positives = [tuple(e) for e in edges_by_id[row['attack_id']]]
        if not positives:
            continue

        key = (row['category'], row['instance'])
        if key not in graph_cache:
            try:
                graph_cache[key] = reload_instance(*key)
            except Exception as e:
                print(f"  [skip {key}: {e}]")
                continue
        G, gt = graph_cache[key]

        pred = detector_for(row['attacker'])(G)
        n_neg = len(positives) * NEG_PER_POS
        negatives = sample_negatives(G, positives, n_neg, rng)

        for u, v in positives:
            feats = edge_features(G, u, v, pred, mu=MU_FOR_FEATS)
            feats.update({'y': 1, 'attacker': row['attacker'],
                          'category': row['category'],
                          'instance': row['instance'],
                          'attack_id': row['attack_id']})
            all_records.append(feats)
        for u, v in negatives:
            feats = edge_features(G, u, v, pred, mu=MU_FOR_FEATS)
            feats.update({'y': 0, 'attacker': row['attacker'],
                          'category': row['category'],
                          'instance': row['instance'],
                          'attack_id': row['attack_id']})
            all_records.append(feats)

    return pd.DataFrame(all_records)


# ----------------------------------------------------------------------
#  Modeling
# ----------------------------------------------------------------------

FEATURE_COLS = [
    'deg_u', 'deg_v', 'deg_min', 'deg_max',
    'common_neighbors', 'jaccard', 'adamic_adar',
    'same_pred_comm', 'cc_u', 'cc_v', 'tri_u', 'tri_v',
    'mu_triad_part_u', 'mu_triad_part_v',
    'boundary_u', 'boundary_v',
]


def fit_models(df, label="ALL"):
    X = df[FEATURE_COLS].values.astype(float)
    y = df['y'].values
    if len(set(y)) < 2:
        print(f"[{label}] only one class — skipping")
        return

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=RANDOM_SEED,
                                          stratify=y)

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    # ----- logistic regression -----
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_s, ytr)
    auc_lr = roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1])

    # ----- gradient boosting -----
    gb = GradientBoostingClassifier(random_state=RANDOM_SEED).fit(Xtr, ytr)
    auc_gb = roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])

    print(f"\n[{label}]  n={len(df)}  pos={int(y.sum())}  "
          f"AUC(LR)={auc_lr:.3f}  AUC(GB)={auc_gb:.3f}")

    print("  Logistic regression coefficients (standardized):")
    for name, w in sorted(zip(FEATURE_COLS, lr.coef_[0]),
                          key=lambda kv: -abs(kv[1])):
        print(f"    {name:<22} {w:+.3f}")

    print("  Gradient boosting feature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, gb.feature_importances_),
                            key=lambda kv: -kv[1]):
        print(f"    {name:<22} {imp:.3f}")


# ----------------------------------------------------------------------
#  Cross-attacker comparison
# ----------------------------------------------------------------------

def compare_attackers(df):
    """Mean of each feature for positive edges, sliced by attacker."""
    pos = df[df['y'] == 1]
    print("\n=== mean feature value of POSITIVE (chosen) edges per attacker ===")
    summary = pos.groupby('attacker')[FEATURE_COLS].mean()
    print(summary.round(3).T)


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    df = build_dataset()
    print(f"Built dataset: {len(df)} rows, "
          f"{df['y'].sum()} positives, "
          f"{(df['y']==0).sum()} negatives")
    df.to_csv("results/feature_dataset.csv", index=False)

    fit_models(df, label="ALL ATTACKERS COMBINED")
    for atk in ['SA-Q', 'SA-CPM', 'muMEA-best']:
        sub = df[df['attacker'] == atk]
        if not sub.empty:
            fit_models(sub, label=atk)

    compare_attackers(df)


if __name__ == "__main__":
    main()