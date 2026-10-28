"""Top-K selection: QUBO + simulated annealing vs greedy vs exact enumeration.

Question: when does a "quantum-inspired" formulation (QUBO solved by simulated
annealing) actually beat a plain greedy algorithm?

Problem: pick exactly K items out of n to maximise
    V(S) = sum_{i in S} a_i  -  sum_{i<j in S} b_ij
a_i  = value of item i, b_ij = overlap (redundancy) between items i and j.

QUBO form (minimised by the annealer):
    E(x) = -a.x + 1/2 x^T B x + P (sum(x) - K)^2,   x in {0,1}^n
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

import numpy as np


def make_instance(n: int, seed: int, density: float = 0.0, max_overlap: float = 0.0):
    """Random instance. density = 0 or max_overlap = 0 gives a separable problem."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 21, size=n).astype(float)
    b = np.zeros((n, n))
    if density > 0 and max_overlap > 0:
        mask = rng.random((n, n)) < density
        vals = rng.uniform(0.0, max_overlap, size=(n, n))
        b = np.triu(mask * vals, 1)
        b = b + b.T
    return a, b


def value(sel, a, b) -> float:
    idx = list(sel)
    if not idx:
        return 0.0
    return float(a[idx].sum() - b[np.ix_(idx, idx)].sum() / 2.0)


def greedy(a, b, k: int):
    chosen: list[int] = []
    gain = a.copy()
    for _ in range(k):
        cand = [i for i in range(len(a)) if i not in chosen]
        best = max(cand, key=lambda i: (gain[i], -i))
        chosen.append(best)
        gain = gain - b[best]
    return tuple(sorted(chosen))


def exact(a, b, k: int):
    best_sel, best_v = None, -math.inf
    for sel in itertools.combinations(range(len(a)), k):
        v = value(sel, a, b)
        if v > best_v + 1e-12:
            best_sel, best_v = sel, v
    return best_sel


def anneal(a, b, k: int, sweeps: int = 2000, t1: float = 0.01, seed: int = 0):
    """Single-bit-flip Metropolis annealing on the QUBO energy.

    The penalty is the smallest safe one: one item more or less than k never
    pays. The starting temperature is twice the penalty, so the walk can cross
    infeasible states early on. Returns the best *feasible* selection seen.
    """
    n = len(a)
    rnd = random.Random(seed)
    penalty = float(max(a.max(), b.sum(axis=1).max()) + 1.0)
    t0 = 2.0 * penalty
    x = [0] * n
    for i in rnd.sample(range(n), k):
        x[i] = 1
    bx = b @ np.array(x, dtype=float)  # (Bx)_i, b has zero diagonal
    s = sum(x)
    best_sel = tuple(j for j in range(n) if x[j])
    best_v = value(best_sel, a, b)

    def pen(m):
        return penalty * (m - k) ** 2

    steps = sweeps * n
    for step in range(steps):
        t = t0 * (t1 / t0) ** (step / max(1, steps - 1))
        i = rnd.randrange(n)
        if x[i] == 0:
            delta = -a[i] + bx[i] + pen(s + 1) - pen(s)
        else:
            delta = a[i] - bx[i] + pen(s - 1) - pen(s)
        if delta <= 0 or rnd.random() < math.exp(-delta / t):
            sign = 1 if x[i] == 0 else -1
            x[i] = 1 - x[i]
            s += sign
            bx += sign * b[:, i]
            if s == k:
                sel = tuple(j for j in range(n) if x[j])
                v = value(sel, a, b)
                if v > best_v + 1e-12:
                    best_sel, best_v = sel, v
    return best_sel


def anneal_swap(a, b, k: int, sweeps: int = 2000, t1: float = 0.01, seed: int = 0):
    """Added after the first run (see CRITERIA.md, v2): annealing that swaps one
    chosen item for one unchosen item, so every state has exactly k items.
    This is no longer a QUBO: the constraint is enforced by the move, not by a penalty.
    """
    n = len(a)
    rnd = random.Random(seed)
    sel = set(rnd.sample(range(n), k))
    cur_v = value(sel, a, b)
    best_sel, best_v = tuple(sorted(sel)), cur_v
    t0 = float(a.max())
    steps = sweeps * n
    for step in range(steps):
        t = t0 * (t1 / t0) ** (step / max(1, steps - 1))
        out_i = rnd.choice(tuple(sel))
        in_i = rnd.choice([j for j in range(n) if j not in sel])
        cand = (sel - {out_i}) | {in_i}
        v = value(cand, a, b)
        if v >= cur_v or rnd.random() < math.exp((v - cur_v) / t):
            sel, cur_v = cand, v
            if v > best_v + 1e-12:
                best_sel, best_v = tuple(sorted(sel)), v
    return best_sel


@dataclass
class Row:
    regime: str
    instances: int
    greedy_optimal: int
    anneal_optimal: int
    greedy_mean_gap: float
    anneal_mean_gap: float
    swap_optimal: int
    swap_mean_gap: float


def run(regime: str, n: int, k: int, instances: int, density: float, max_overlap: float) -> Row:
    g_ok = a_ok = s_ok = 0
    g_gap = a_gap = s_gap = 0.0
    for seed in range(instances):
        a, b = make_instance(n, seed, density, max_overlap)
        v_opt = value(exact(a, b, k), a, b)
        v_g = value(greedy(a, b, k), a, b)
        v_a = value(anneal(a, b, k, seed=seed), a, b)
        v_s = value(anneal_swap(a, b, k, sweeps=200, seed=seed), a, b)
        g_ok += abs(v_g - v_opt) < 1e-9
        a_ok += abs(v_a - v_opt) < 1e-9
        s_ok += abs(v_s - v_opt) < 1e-9
        g_gap += (v_opt - v_g) / v_opt if v_opt else 0.0
        a_gap += (v_opt - v_a) / v_opt if v_opt else 0.0
        s_gap += (v_opt - v_s) / v_opt if v_opt else 0.0
    return Row(regime, instances, g_ok, a_ok, g_gap / instances, a_gap / instances,
               s_ok, s_gap / instances)


def main() -> None:
    n, k, instances = 20, 5, 100
    rows = [
        run("separable (no overlap)", n, k, instances, 0.0, 0.0),
        run("interacting (30% pairs overlap, up to 15)", n, k, instances, 0.3, 15.0),
    ]
    lines = [
        f"n = {n}, K = {k}, {instances} instances per regime, seeds 0..{instances - 1}",
        "",
        "Optimal = same value as exact enumeration. Gap = mean shortfall vs the optimum.",
        "",
        "| Regime | Method | Optimal | Mean gap |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        lines += [
            f"| {r.regime} | Greedy | {r.greedy_optimal}/{r.instances} | {r.greedy_mean_gap:.2%} |",
            f"| {r.regime} | QUBO + bit-flip annealing (v1) | {r.anneal_optimal}/{r.instances} | {r.anneal_mean_gap:.2%} |",
            f"| {r.regime} | Swap annealing, not a QUBO (v2) | {r.swap_optimal}/{r.instances} | {r.swap_mean_gap:.2%} |",
        ]
    out = "\n".join(lines)
    print(out)
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# Results\n\n" + out + "\n")


if __name__ == "__main__":
    main()
