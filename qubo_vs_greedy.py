"""Top-K selection: QUBO + simulated annealing vs greedy vs exact enumeration.

Question: when does a "quantum-inspired" formulation (QUBO solved by simulated
annealing) actually beat a plain greedy algorithm?

Problem: pick exactly K items out of n to maximise
    V(S) = sum_{i in S} a_i  -  sum_{i<j in S} b_ij
a_i  = value of item i, b_ij = overlap (redundancy) between items i and j.

QUBO form (minimised by the annealer):
    E(x) = -a.x + 1/2 x^T B x + P (sum(x) - K)^2,   x in {0,1}^n

Running this file regenerates the v1/v2 table (RESULTS.md, deterministic) and
its timings (TIMINGS.md, machine-dependent). The v3 study is study_v3.py.
"""
from __future__ import annotations

import bisect
import functools
import itertools
import math
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Safety margin for the fast paths below. The incremental quantities they use
# differ from value() by rounding only (about 1e-12 here), far below this margin.
# Whenever a decision falls inside the margin, the code recomputes value() and
# decides exactly as the v1/v2 code did, so the selections are unchanged.
_TOL = 1e-7


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


def swap_delta(al, rows, bx, out_i: int, in_i: int) -> float:
    """Change of V when out_i (chosen) is swapped for in_i (not chosen).

    bx[j] = sum_{i in S} b_ij for the current selection S (b has zero diagonal).
    Removing out_i loses a_out but gives back its overlaps with the rest of S
    (= bx[out_i]); adding in_i gains a_in and pays its overlaps with S minus out_i
    (= bx[in_i] - b[in_i][out_i]).
    """
    return al[in_i] - al[out_i] + bx[out_i] - bx[in_i] + rows[in_i][out_i]


def greedy(a, b, k: int):
    chosen: list[int] = []
    gain = a.copy()
    for _ in range(k):
        cand = [i for i in range(len(a)) if i not in chosen]
        best = max(cand, key=lambda i: (gain[i], -i))
        chosen.append(best)
        gain = gain - b[best]
    return tuple(sorted(chosen))


def local_search(a, b, start):
    """First-improvement 1-swap hill climbing from `start` (deterministic).

    Scans chosen items in increasing order, and for each the unchosen items in
    increasing order; applies the first swap that raises V by more than 1e-9 and
    restarts the scan. Stops at a 1-swap local optimum.
    Returns (selection, number of swaps evaluated).
    """
    n = len(a)
    al, rows = a.tolist(), b.tolist()
    chosen = sorted(start)
    inside = [False] * n
    for i in chosen:
        inside[i] = True
    bx = [sum(rows[j][i] for i in chosen) for j in range(n)]
    evaluated = 0
    improved = True
    while improved:
        improved = False
        for out_i in chosen:
            for in_i in range(n):
                if inside[in_i]:
                    continue
                evaluated += 1
                if swap_delta(al, rows, bx, out_i, in_i) > 1e-9:
                    chosen.remove(out_i)
                    bisect.insort(chosen, in_i)
                    inside[out_i], inside[in_i] = False, True
                    bx = [sum(rows[j][i] for i in chosen) for j in range(n)]
                    improved = True
                    break
            if improved:
                break
    return tuple(chosen), evaluated


def greedy_ls(a, b, k: int):
    """Greedy, then 1-swap local search (v3 classical baseline)."""
    return local_search(a, b, greedy(a, b, k))[0]


def _exact_loop(a, b, k: int):
    """The v1/v2 enumeration, kept as the reference for exact()."""
    best_sel, best_v = None, -math.inf
    for sel in itertools.combinations(range(len(a)), k):
        v = value(sel, a, b)
        if v > best_v + 1e-12:
            best_sel, best_v = sel, v
    return best_sel


@functools.lru_cache(maxsize=4)
def _subsets(n: int, k: int):
    return np.array(list(itertools.combinations(range(n), k)), dtype=np.intp)


def exact(a, b, k: int):
    """Exact enumeration of all C(n, k) subsets, vectorised with numpy.

    Returns the same subset as the v1/v2 loop (_exact_loop): the subsets are
    scored in the same lexicographic order, and every subset whose vectorised
    score is within _TOL of beating the running best is re-scored with value()
    and judged by the loop's own rule.
    """
    n = len(a)
    if k <= 0 or k > n or math.comb(n, k) > 2_000_000:
        return _exact_loop(a, b, k)
    subsets = _subsets(n, k)
    approx = a[subsets].sum(axis=1)
    for p in range(k):
        for q in range(p + 1, k):
            approx -= b[subsets[:, p], subsets[:, q]]
    best_sel, best_v, pos = None, -math.inf, 0
    while True:
        hits = np.flatnonzero(approx[pos:] > best_v + 1e-12 - _TOL)
        for h in hits:
            sel = tuple(subsets[pos + h].tolist())
            v = value(sel, a, b)
            if v > best_v + 1e-12:
                best_sel, best_v, pos = sel, v, pos + h + 1
                break
        else:
            return best_sel


def anneal(a, b, k: int, sweeps: int = 2000, t1: float = 0.01, seed: int = 0,
           penalty_scale: float = 1.0, t0_scale: float = 2.0):
    """Single-bit-flip Metropolis annealing on the QUBO energy.

    The penalty is the smallest safe one: one item more or less than k never
    pays. The starting temperature is twice the penalty, so the walk can cross
    infeasible states early on. Returns the best *feasible* selection seen.
    Budget: sweeps * n proposed flips.

    penalty_scale and t0_scale (v3 sensitivity study) multiply the penalty and
    set T0 = t0_scale * penalty; the defaults are the v1 settings.
    The arithmetic of each step is the v1 arithmetic, in the same order, on the
    same random draws, so the result is identical to v1.
    """
    n = len(a)
    rnd = random.Random(seed)
    penalty = float(max(a.max(), b.sum(axis=1).max()) + 1.0) * penalty_scale
    t0 = t0_scale * penalty
    x = [0] * n
    for i in rnd.sample(range(n), k):
        x[i] = 1
    bx = (b @ np.array(x, dtype=float)).tolist()  # (Bx)_i, b has zero diagonal
    s = sum(x)
    best_sel = tuple(j for j in range(n) if x[j])
    best_v = value(best_sel, a, b)
    al, cols = a.tolist(), b.T.tolist()
    pen = [penalty * (m - k) ** 2 for m in range(n + 2)]
    ratio, steps = t1 / t0, sweeps * n
    den = max(1, steps - 1)
    randrange, uniform = rnd.randrange, rnd.random
    for sweep in range(sweeps):
        # value of the current x (feasible or not), only used to skip value()
        # calls that cannot improve the best; recomputed every sweep.
        xa = np.array(x, dtype=float)
        cur_v = float(a @ xa - 0.5 * (xa @ b @ xa))
        for step in range(sweep * n, sweep * n + n):
            i = randrange(n)
            if x[i] == 0:
                delta = -al[i] + bx[i] + pen[s + 1] - pen[s]
            else:
                delta = al[i] - bx[i] + pen[s - 1] - pen[s]
            if delta <= 0 or uniform() < math.exp(-delta / (t0 * ratio ** (step / den))):
                gain = al[i] - bx[i]
                if x[i] == 0:
                    x[i], s, cur_v = 1, s + 1, cur_v + gain
                    bx = [u + w for u, w in zip(bx, cols[i])]
                else:
                    x[i], s, cur_v = 0, s - 1, cur_v - gain
                    bx = [u - w for u, w in zip(bx, cols[i])]
                if s == k and cur_v > best_v + 1e-12 - _TOL:
                    sel = tuple(j for j in range(n) if x[j])
                    v = value(sel, a, b)
                    cur_v = v
                    if v > best_v + 1e-12:
                        best_sel, best_v = sel, v
    return best_sel


def anneal_swap(a, b, k: int, sweeps: int = 2000, t1: float = 0.01, seed: int = 0):
    """Added after the first run (see CRITERIA.md, v2): annealing that swaps one
    chosen item for one unchosen item, so every state has exactly k items.
    This is no longer a QUBO: the constraint is enforced by the move, not by a penalty.
    Budget: sweeps * n proposed swaps.

    Each proposal is scored incrementally with swap_delta (O(1)) instead of
    value(). Accept/reject decisions within _TOL of the v2 thresholds are
    re-decided with value(), and the random draws are taken in the v2 order, so
    the result is identical to v2.
    """
    n = len(a)
    rnd = random.Random(seed)
    sel = set(rnd.sample(range(n), k))
    cur_v = value(sel, a, b)
    best_sel, best_v = tuple(sorted(sel)), cur_v
    # v2 used t0 = a.max(); the floor at t1 only matters when every a_i <= t1
    # (e.g. the blank control, where v2 divided by zero), never in the benchmark.
    t0 = max(float(a.max()), t1)
    ratio, steps = t1 / t0, sweeps * n
    den = max(1, steps - 1)
    al, rows = a.tolist(), b.tolist()
    rest = [j for j in range(n) if j not in sel]  # unchosen items, increasing
    choice, uniform = rnd.choice, rnd.random
    for sweep in range(sweeps):
        # incremental state, recomputed every sweep so rounding cannot build up
        bx = [sum(rows[j][i] for i in sel) for j in range(n)]
        cur_v = sum(al[i] for i in sel) - 0.5 * sum(bx[i] for i in sel)
        for step in range(sweep * n, sweep * n + n):
            out_i = choice(tuple(sel))
            in_i = choice(rest)
            d = swap_delta(al, rows, bx, out_i, in_i)
            cand = None
            if d > _TOL:  # surely better: v2 accepts without a random draw
                accept = True
            elif d < -_TOL:  # surely worse: v2 draws u and compares with exp(d / t)
                t = t0 * ratio ** (step / den)
                u, p = uniform(), math.exp(d / t)
                if abs(u - p) > _TOL:
                    accept = u < p
                else:
                    cand = (sel - {out_i}) | {in_i}
                    accept = u < math.exp((value(cand, a, b) - value(sel, a, b)) / t)
            else:  # near tie: decide exactly as v2
                cand = (sel - {out_i}) | {in_i}
                v, v_sel = value(cand, a, b), value(sel, a, b)
                accept = v >= v_sel or uniform() < math.exp(
                    (v - v_sel) / (t0 * ratio ** (step / den)))
            if accept:
                sel = cand if cand is not None else (sel - {out_i}) | {in_i}
                rest.remove(in_i)
                bisect.insort(rest, out_i)
                ri, ro = rows[in_i], rows[out_i]
                bx = [u + p - q for u, p, q in zip(bx, ri, ro)]
                cur_v += d
                if cur_v > best_v + 1e-12 - _TOL:
                    v = value(sel, a, b)
                    cur_v = v
                    if v > best_v + 1e-12:
                        best_sel, best_v = tuple(sorted(sel)), v
    return best_sel


def milp_optimum(a, b, k: int, time_limit: float = 120.0):
    """Exact reference where enumeration is impossible (v3): an MILP solved by
    HiGHS through scipy.optimize.milp.

        maximise  a.x - sum_{i<j, b_ij > 0} b_ij y_ij
        s.t.      sum(x) = k,  y_ij >= x_i + x_j - 1,  y_ij >= 0,  x binary.

    b >= 0, so at the optimum y_ij = max(0, x_i + x_j - 1) = x_i x_j and the
    objective is V(x). Returns (selection, proven): proven is True only when
    HiGHS reports an optimal solution (relative gap 0, HiGHS's default absolute
    gap 1e-6) within the time limit; otherwise the selection is the best one
    found (or None).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if (b < 0).any():
        raise ValueError("the MILP linearisation needs b >= 0")
    n = len(a)
    ii, jj = np.nonzero(np.triu(b, 1))
    m = len(ii)
    c = np.concatenate([-a, b[ii, jj]])  # milp minimises
    pair_rows = coo_matrix(
        (np.tile([1.0, -1.0, -1.0], m),
         (np.repeat(np.arange(m), 3), np.stack([n + np.arange(m), ii, jj], 1).ravel())),
        shape=(m, n + m))
    card_row = coo_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(n))), shape=(1, n + m))
    constraints = [LinearConstraint(card_row, k, k)]
    if m:
        constraints.append(LinearConstraint(pair_rows, -1.0, np.inf))
    res = milp(c, constraints=constraints,
               integrality=np.concatenate([np.ones(n), np.zeros(m)]),
               bounds=Bounds(0.0, 1.0),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return None, False
    sel = tuple(int(i) for i in np.flatnonzero(res.x[:n] > 0.5))
    return sel, bool(res.status == 0 and len(sel) == k)


def wilson(successes: int, trials: int, z: float = 1.959964):
    """Wilson score 95 % interval for a success proportion."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return max(0.0, centre - half), min(1.0, centre + half)


def machine_info() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"{cpu}, {os.cpu_count()} logical CPUs, {platform.system()}, "
            f"Python {platform.python_version()}, numpy {np.__version__}, one process")


def fmt_int(x: int) -> str:
    """40000 -> '40 000' (the README style)."""
    return f"{x:,}".replace(",", " ")


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t) * 1000.0


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
    ms: dict = field(default_factory=dict)  # method -> total ms over the instances


# v1/v2 budgets, in proposals evaluated per run: sweeps * n
V1_SWEEPS = 2000  # QUBO bit-flip annealing: 2000 * 20 = 40 000 proposed flips
V2_SWEEPS = 200   # swap annealing: 200 * 20 = 4 000 proposed swaps


def run(regime: str, n: int, k: int, instances: int, density: float, max_overlap: float) -> Row:
    g_ok = a_ok = s_ok = 0
    g_gap = a_gap = s_gap = 0.0
    ms = dict.fromkeys(("exact", "greedy", "anneal", "swap"), 0.0)
    for seed in range(instances):
        a, b = make_instance(n, seed, density, max_overlap)
        sel_opt, t_e = timed(exact, a, b, k)
        sel_g, t_g = timed(greedy, a, b, k)
        sel_a, t_a = timed(anneal, a, b, k, sweeps=V1_SWEEPS, seed=seed)
        sel_s, t_s = timed(anneal_swap, a, b, k, sweeps=V2_SWEEPS, seed=seed)
        for key, t in zip(ms, (t_e, t_g, t_a, t_s)):
            ms[key] += t
        v_opt = value(sel_opt, a, b)
        v_g, v_a, v_s = value(sel_g, a, b), value(sel_a, a, b), value(sel_s, a, b)
        g_ok += abs(v_g - v_opt) < 1e-9
        a_ok += abs(v_a - v_opt) < 1e-9
        s_ok += abs(v_s - v_opt) < 1e-9
        g_gap += (v_opt - v_g) / v_opt if v_opt else 0.0
        a_gap += (v_opt - v_a) / v_opt if v_opt else 0.0
        s_gap += (v_opt - v_s) / v_opt if v_opt else 0.0
    return Row(regime, instances, g_ok, a_ok, g_gap / instances, a_gap / instances,
               s_ok, s_gap / instances, ms)


def main() -> None:
    n, k, instances = 20, 5, 100
    t_start = time.perf_counter()
    rows = [
        run("separable (no overlap)", n, k, instances, 0.0, 0.0),
        run("interacting (30% pairs overlap, up to 15)", n, k, instances, 0.3, 15.0),
    ]
    total_s = time.perf_counter() - t_start
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
    lines += [
        "",
        "**Unequal budgets.** Budget = proposals evaluated per run. The QUBO bit-flip annealer "
        f"gets {V1_SWEEPS} sweeps x {n} = {fmt_int(V1_SWEEPS * n)} proposed flips; the swap annealer gets "
        f"{V2_SWEEPS} sweeps x {n} = {fmt_int(V2_SWEEPS * n)} proposed swaps, 10 times fewer. The gap favours "
        "the QUBO annealer, which still loses. An equal-budget comparison, with 5 seeds per instance "
        "and a greedy + local search baseline, is in RESULTS_V3.md. Timings: TIMINGS.md.",
    ]
    out = "\n".join(lines)
    print(out)
    (HERE / "RESULTS.md").write_text("# Results\n\n" + out + "\n", encoding="utf-8")

    labels = (("exact", "Exact enumeration (C(20,5) = 15 504 subsets)"),
              ("greedy", "Greedy"),
              ("anneal", f"QUBO + bit-flip annealing (v1), {fmt_int(V1_SWEEPS * n)} flips"),
              ("swap", f"Swap annealing (v2), {fmt_int(V2_SWEEPS * n)} swaps"))
    tl = [
        "# Timings (v1/v2 table)",
        "",
        "Mean wall-clock time per instance, in milliseconds. Indicative only: measured on "
        f"{machine_info()}. These numbers change from run to run and machine to machine, "
        "so they are kept out of RESULTS.md and are not checked by CI.",
        "",
        "| Regime | Method | Mean ms per instance |",
        "|---|---|---:|",
    ]
    for r in rows:
        for key, label in labels:
            tl.append(f"| {r.regime} | {label} | {r.ms[key] / r.instances:.1f} |")
    tl += ["", f"Whole script: {total_s:.1f} s."]
    timings = "\n".join(tl)
    print("\n" + timings)
    (HERE / "TIMINGS.md").write_text(timings + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
