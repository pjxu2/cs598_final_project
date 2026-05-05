"""
Objective functions for the SA attacker. Convention: HIGHER score = MORE
damage to community detection. SA in sa_attack.py maximizes whatever it gets.

Available objectives:
  - make_obj_ari_drop       : maximize -ARI(detect(H), gt)
  - make_obj_modularity_drop: maximize Q(G0) - Q(H) under modularity clustering
  - make_obj_contrastive    : damage one detector while preserving another
                              (useful for studying transfer / specificity)
"""

from sklearn.metrics import adjusted_rand_score
import networkx as nx
import community as community_louvain  # python-louvain, only used for Q


# ---------- helpers ----------

def _ari(detect_fn, H, gt_lst, nodes_order):
    pred = detect_fn(H)
    return adjusted_rand_score(gt_lst, [pred[v] for v in nodes_order])


# ---------- objective factories ----------

def make_obj_ari_drop(gt, detect_fn):
    """f(H) = -ARI(detect_fn(H), gt). Maximizing means low ARI = damaged."""
    nodes_order = sorted(gt.keys())
    gt_lst = [gt[v] for v in nodes_order]

    def f(H):
        return -_ari(detect_fn, H, gt_lst, nodes_order)
    return f


def make_obj_modularity_drop(G0, detect_fn):
    """
    f(H) = Q(G0, partition_on_G0) - Q(H, partition_on_H).
    Larger = attack reduced the modularity score more.
    Note Q is computed *under that graph's own clustering*, which matches
    how Hsu et al. report "modularity reduction".
    """
    pred0 = detect_fn(G0)
    Q0 = community_louvain.modularity(pred0, G0)

    def f(H):
        pred_h = detect_fn(H)
        Qh = community_louvain.modularity(pred_h, H)
        return Q0 - Qh
    return f


def make_obj_contrastive(gt, detect_fn_target, detect_fn_avoid, alpha=1.0):
    """
    Damage `target` detector while preserving `avoid` detector.
    f(H) = -ARI_target(H) + alpha * ARI_avoid(H).
    """
    nodes_order = sorted(gt.keys())
    gt_lst = [gt[v] for v in nodes_order]

    def f(H):
        ari_t = _ari(detect_fn_target, H, gt_lst, nodes_order)
        ari_a = _ari(detect_fn_avoid,  H, gt_lst, nodes_order)
        return -ari_t + alpha * ari_a
    return f