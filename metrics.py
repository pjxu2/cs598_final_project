"""
Post-hoc evaluation: given an original graph, an attacked graph, ground-truth
labels, and a detection function, report ARI/NMI before vs. after the attack.

This is for *measuring* attack quality. To *drive* an attack, use
objectives.py instead.
"""

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _aligned(label_dict, nodes):
    """Pull labels out of a {node: label} dict in a fixed node order."""
    return [label_dict[v] for v in nodes]


def eval_attack(G_orig, G_attacked, gt, detect_fn):
    """
    Run detect_fn on both graphs and compare to ground truth.

    Args:
        G_orig    : original graph
        G_attacked: graph after edge modifications
        gt        : {node: community_id} ground truth
        detect_fn : callable(G) -> {node: community_id}, e.g. louvain_q

    Returns dict with ari_before/after and nmi_before/after.
    """
    nodes = list(G_orig.nodes)
    pred_orig = detect_fn(G_orig)
    pred_att  = detect_fn(G_attacked)

    gt_lst        = _aligned(gt,        nodes)
    pred_orig_lst = _aligned(pred_orig, nodes)
    pred_att_lst  = _aligned(pred_att,  nodes)

    return {
        'ari_before': adjusted_rand_score(gt_lst, pred_orig_lst),
        'ari_after':  adjusted_rand_score(gt_lst, pred_att_lst),
        'nmi_before': normalized_mutual_info_score(gt_lst, pred_orig_lst),
        'nmi_after':  normalized_mutual_info_score(gt_lst, pred_att_lst),
    }