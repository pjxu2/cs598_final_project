"""
Simulated annealing attacker.

Maintains G_work = G + added - removed. At each step we propose add/remove/swap
and Metropolis accept based on objective_fn (HIGHER = more damage).

Two constraints:
  - budget         : max # of net modifications allowed
  - k_per_vertex   : "k-unnoticeability" cap. No vertex may end up with more
                     than k newly added edges incident. Matches muMEA's k
                     hyperparameter so we can compare fairly. None disables it.
"""

import math
import random
from collections import defaultdict


def _propose_add(G_work, nodes, added, added_per_node,
                 k_per_vertex, max_tries=20):
    for _ in range(max_tries):
        u, v = random.sample(nodes, 2)
        e = (min(u, v), max(u, v))
        if G_work.has_edge(u, v) or e in added:
            continue
        if k_per_vertex is not None and (
            added_per_node[u] >= k_per_vertex
            or added_per_node[v] >= k_per_vertex
        ):
            continue
        G_work.add_edge(*e)
        added.add(e)
        added_per_node[u] += 1
        added_per_node[v] += 1

        def undo(e=e, u=u, v=v):
            G_work.remove_edge(*e)
            added.discard(e)
            added_per_node[u] -= 1
            added_per_node[v] -= 1
        return e, undo
    return None, None


def _propose_remove(G_work, added, removed):
    candidates = [
        (min(u, v), max(u, v)) for u, v in G_work.edges()
        if (min(u, v), max(u, v)) not in added
        and (min(u, v), max(u, v)) not in removed
    ]
    if not candidates:
        return None, None
    e = random.choice(candidates)
    G_work.remove_edge(*e)
    removed.add(e)

    def undo():
        G_work.add_edge(*e)
        removed.discard(e)
    return e, undo


def _propose_swap(G_work, nodes, added, added_per_node,
                  k_per_vertex, max_tries=20):
    if not added:
        return None, None
    old = random.choice(list(added))
    G_work.remove_edge(*old)
    added.discard(old)
    added_per_node[old[0]] -= 1
    added_per_node[old[1]] -= 1

    for _ in range(max_tries):
        u, v = random.sample(nodes, 2)
        new = (min(u, v), max(u, v))
        if new == old or G_work.has_edge(u, v) or new in added:
            continue
        if k_per_vertex is not None and (
            added_per_node[u] >= k_per_vertex
            or added_per_node[v] >= k_per_vertex
        ):
            continue
        G_work.add_edge(*new)
        added.add(new)
        added_per_node[u] += 1
        added_per_node[v] += 1

        def undo(old=old, new=new):
            G_work.remove_edge(*new)
            added.discard(new)
            added_per_node[new[0]] -= 1
            added_per_node[new[1]] -= 1
            G_work.add_edge(*old)
            added.add(old)
            added_per_node[old[0]] += 1
            added_per_node[old[1]] += 1
        return (old, new), undo

    # couldn't find a replacement; restore old and bail
    G_work.add_edge(*old)
    added.add(old)
    added_per_node[old[0]] += 1
    added_per_node[old[1]] += 1
    return None, None


def sa_attack(G, budget, objective_fn,
              allow_deletions=False,
              k_per_vertex=None,
              n_iter=3000, T0=0.05, T_end=0.001,
              verbose=False):
    """
    Args:
        G              : NetworkX graph
        budget         : max # of net modifications
        objective_fn   : callable(H) -> scalar, higher = more damage
        allow_deletions: if True, SA may also remove existing edges
        k_per_vertex   : per-vertex cap on newly added edges (None to disable)
        n_iter         : number of MCMC steps
        T0, T_end      : initial / final temperature (geometric cooling)

    Returns:
        added, removed, best_score, history
    """
    G_work = G.copy()
    nodes = list(G.nodes)
    added, removed = set(), set()
    added_per_node = defaultdict(int)

    current = objective_fn(G_work)
    best_added, best_removed, best_score = set(), set(), current
    history = [current]

    T = T0
    cooling = (T_end / T0) ** (1.0 / max(n_iter, 1))

    for it in range(n_iter):
        moves = ['swap']
        if len(added) + len(removed) < budget:
            moves.append('add')
            if allow_deletions:
                moves.append('remove')
        move = random.choice(moves)

        if move == 'add':
            _, undo = _propose_add(G_work, nodes, added,
                                   added_per_node, k_per_vertex)
        elif move == 'remove':
            _, undo = _propose_remove(G_work, added, removed)
        else:  # swap
            _, undo = _propose_swap(G_work, nodes, added,
                                    added_per_node, k_per_vertex)

        if undo is None:
            history.append(current)
            T *= cooling
            continue

        new_score = objective_fn(G_work)
        delta = new_score - current

        if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-12)):
            current = new_score
            if current > best_score:
                best_added   = set(added)
                best_removed = set(removed)
                best_score   = current
        else:
            undo()

        history.append(current)
        T *= cooling

        if verbose and it % 200 == 0:
            print(f"  iter {it:5d}  T={T:.4f}  cur={current:+.4f}  best={best_score:+.4f}")

    return best_added, best_removed, best_score, history