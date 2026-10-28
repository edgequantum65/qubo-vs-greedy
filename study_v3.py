"""v3 study, pre-registered in CRITERIA.md (section v3) before it was run.

1. n = 20, K = 5, interacting regime, the 100 v1/v2 instances: greedy,
   greedy + 1-swap local search, QUBO bit-flip annealing and swap annealing,
   the two annealers at equal budgets (200 n and 2000 n proposals per run),
   5 seeds per instance. Reference: exact enumeration.
2. QUBO sensitivity: penalty x {1, 2, 5} of the minimal safe value,
   T0 x {0.1, 0.5, 2} of the penalty, 2000 n proposals, seed r = 0.
3. The MILP reference checked against exact enumeration on the 200 n = 20
   instances (both regimes).
4. n = 60, K = 10 and n = 100, K = 15, interacting regime, 40 instances each:
   same methods; reference = MILP optimum, counted only where HiGHS proved it.

Writes RESULTS_V3.md (quality numbers) and TIMINGS_V3.md (timings, machine-
dependent). Takes about 15 minutes on one core.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from qubo_vs_greedy import (HERE, anneal, anneal_swap, exact, fmt_int, greedy, local_search,
                            machine_info, make_instance, milp_optimum, timed, value, wilson)

SEEDS = 5                    # runs per instance for the annealers; seed = instance + 1000 r
BUDGET_SWEEPS = (200, 2000)  # proposals per run = sweeps * n
DENSITY, OVERLAP = 0.3, 15.0
LARGE = ((60, 10, 40), (100, 15, 40))  # (n, K, instances)
MILP_TIME_LIMIT = 120.0      # seconds per instance
PENALTY_SCALES = (1.0, 2.0, 5.0)
T0_SCALES = (0.1, 0.5, 2.0)
SAME_ENUM, SAME_MILP = 1e-9, 1e-6  # "same value" tolerance vs each reference


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Stat:
    label: str
    proposals: str
    seeds: int
    ok_per_seed: list = field(default_factory=list)  # optimal count per seed (proven instances)
    gaps: list = field(default_factory=list)
    ms: list = field(default_factory=list)
    evaluated: list = field(default_factory=list)    # local search only
    above_reference: int = 0                          # must stay 0
    unproven_runs: int = 0
    unproven_match: int = 0                           # matched or beat the MILP incumbent

    def __post_init__(self):
        self.ok_per_seed = [0] * self.seeds


def method_list(n: int, k: int):
    """(label, proposals text, seeds, run(a, b, seed) -> (selection, evaluated or None))."""
    out = [
        ("Greedy", "n/a", 1, lambda a, b, s: (greedy(a, b, k), None)),
        ("Greedy + 1-swap local search", "see note", 1,
         lambda a, b, s: local_search(a, b, greedy(a, b, k))),
    ]
    for sw in BUDGET_SWEEPS:
        p = fmt_int(sw * n)
        out += [
            (f"QUBO bit-flip annealing, {p} flips", p, SEEDS,
             lambda a, b, s, sw=sw: (anneal(a, b, k, sweeps=sw, seed=s), None)),
            (f"Swap annealing, {p} swaps", p, SEEDS,
             lambda a, b, s, sw=sw: (anneal_swap(a, b, k, sweeps=sw, seed=s), None)),
        ]
    return out


def evaluate(n, k, instances, reference, same):
    """reference(a, b) -> (reference value, proven). Returns stats and reference info."""
    stats = [Stat(lbl, prop, seeds) for lbl, prop, seeds, _ in method_list(n, k)]
    runners = [fn for _, _, _, fn in method_list(n, k)]
    proven_count, ref_ms = 0, []
    for inst in range(instances):
        a, b = make_instance(n, inst, DENSITY, OVERLAP)
        (v_ref, proven), t = timed(reference, a, b)
        ref_ms.append(t)
        proven_count += proven
        for st, fn in zip(stats, runners):
            for r in range(st.seeds):
                (sel, evaluated), t = timed(fn, a, b, inst + 1000 * r)
                v = value(sel, a, b)
                st.ms.append(t)
                if evaluated is not None:
                    st.evaluated.append(evaluated)
                if proven:
                    st.ok_per_seed[r] += v >= v_ref - same
                    st.gaps.append((v_ref - v) / v_ref)
                    st.above_reference += v > v_ref + same
                else:
                    st.unproven_runs += 1
                    st.unproven_match += v >= v_ref - same
        log(f"  n={n} instance {inst + 1}/{instances} done")
    return stats, proven_count, ref_ms


def quality_table(stats, proven: int):
    lines = ["| Method | Proposals per run | Optimal, mean over seeds | Range over seeds | "
             "Wilson 95 % (all runs) | Mean gap |",
             "|---|---:|---:|---:|---:|---:|"]
    for st in stats:
        mean_ok = sum(st.ok_per_seed) / st.seeds
        lo, hi = wilson(sum(st.ok_per_seed), proven * st.seeds)
        rng = (f"{min(st.ok_per_seed)}-{max(st.ok_per_seed)}" if st.seeds > 1 else "deterministic")
        prop = st.proposals
        if st.evaluated:
            prop = f"{sum(st.evaluated) / len(st.evaluated):.0f} swaps (mean)"
        gap = sum(st.gaps) / len(st.gaps) if st.gaps else float("nan")
        lines.append(f"| {st.label} | {prop} | {mean_ok:g}/{proven} | {rng} | "
                     f"{lo:.1%} - {hi:.1%} | {gap:.2%} |")
    return lines


def timing_table(stats):
    lines = ["| Method | Mean ms per run |", "|---|---:|"]
    lines += [f"| {st.label} | {sum(st.ms) / len(st.ms):.1f} |" for st in stats]
    return lines


def main() -> None:
    t_start = time.perf_counter()
    res = ["# Results v3", "",
           "Pre-registered in [CRITERIA.md](CRITERIA.md), section v3, before this run. "
           "Generated by `python study_v3.py`. Timings: TIMINGS_V3.md.", "",
           "- **Budget** = proposals evaluated per run: one proposed bit flip for the QUBO annealer, "
           "one proposed swap for the swap annealer. Both annealers get the same budgets, "
           f"{' and '.join(f'{s} n' for s in BUDGET_SWEEPS)} proposals (n = number of items).",
           f"- Annealers: {SEEDS} runs per instance, seeds instance + 1000 r, r = 0..{SEEDS - 1} "
           "(r = 0 is the v1/v2 seed). *Optimal, mean over seeds* = number of instances solved "
           "optimally, averaged over the seeds; *range* = worst and best seed. Wilson 95 % = "
           "interval on all runs pooled; runs on the same instance are not independent, so it "
           "is narrower than an instance-level interval.",
           "- Greedy + 1-swap local search: greedy, then the first improving swap (chosen items "
           "and unchosen items scanned in increasing order) until no swap improves; deterministic. "
           "Its proposals column is the mean number of swaps it evaluated.",
           "- Gap = mean shortfall vs the reference optimum, over all runs.", ""]
    tim = ["# Timings v3", "",
           f"Indicative only: measured on {machine_info()}. Mean wall-clock time per run.", ""]

    # 1. n = 20 equal budgets
    n, k, instances = 20, 5, 100
    log("Part 1: n = 20 equal budgets")

    def enum_ref(a, b):
        return value(exact(a, b, k), a, b), True

    stats, proven, ref_ms = evaluate(n, k, instances, enum_ref, SAME_ENUM)
    res += [f"## 1. Equal budgets, n = {n}, K = {k}, interacting regime, the {instances} v1/v2 "
            "instances", "", "Reference: exact enumeration.", ""]
    res += quality_table(stats, proven)
    res += ["", f"Runs above the reference optimum (must be 0): "
            f"{sum(st.above_reference for st in stats)}.", ""]
    tim += [f"## 1. n = {n}, K = {k}", "",
            f"Exact enumeration: {sum(ref_ms) / len(ref_ms):.1f} ms per instance.", ""]
    tim += timing_table(stats) + [""]

    # 2. QUBO sensitivity
    log("Part 2: QUBO sensitivity")
    sweeps = BUDGET_SWEEPS[-1]
    grid = {}
    v_opt = []
    for inst in range(instances):
        a, b = make_instance(n, inst, DENSITY, OVERLAP)
        v_opt.append(value(exact(a, b, k), a, b))
    for ps in PENALTY_SCALES:
        for ts in T0_SCALES:
            ok = 0
            for inst in range(instances):
                a, b = make_instance(n, inst, DENSITY, OVERLAP)
                sel = anneal(a, b, k, sweeps=sweeps, seed=inst, penalty_scale=ps, t0_scale=ts)
                ok += abs(value(sel, a, b) - v_opt[inst]) < SAME_ENUM
            grid[ps, ts] = ok
            log(f"  penalty x{ps:g}, T0 x{ts:g}: done")
    res += [f"## 2. QUBO sensitivity, n = {n}, K = {k}, interacting regime", "",
            f"QUBO bit-flip annealing, {fmt_int(sweeps * n)} proposals, seed r = 0, optimal "
            f"instances out of {instances}. Penalty = multiple of the minimal safe penalty "
            "(max(max a, max row sum of b) + 1); T0 = multiple of the penalty. "
            "The cell (1 x, 2 x) is the v1 setting.", "",
            "| Penalty | T0 = 0.1 x penalty | T0 = 0.5 x penalty | T0 = 2 x penalty |",
            "|---|---:|---:|---:|"]
    for ps in PENALTY_SCALES:
        res.append(f"| {ps:g} x minimal | " + " | ".join(
            f"{grid[ps, ts]}/{instances}" for ts in T0_SCALES) + " |")
    res.append("")

    # 3. MILP vs enumeration at n = 20
    log("Part 3: MILP vs enumeration at n = 20")
    match = proven_milp = 0
    milp_ms = []
    for density, overlap in ((0.0, 0.0), (DENSITY, OVERLAP)):
        for inst in range(instances):
            a, b = make_instance(n, inst, density, overlap)
            (sel, proven), t = timed(milp_optimum, a, b, k, MILP_TIME_LIMIT)
            milp_ms.append(t)
            proven_milp += proven
            match += sel is not None and abs(value(sel, a, b) - value(exact(a, b, k), a, b)) <= SAME_MILP
    res += [f"## 3. MILP reference checked against enumeration, n = {n}, K = {k}", "",
            "MILP (scipy.optimize.milp, HiGHS): maximise a.x - sum b_ij y_ij, y_ij >= x_i + x_j - 1, "
            "y_ij >= 0 for the pairs with b_ij > 0, sum x = K, x binary; relative gap 0.", "",
            f"- Proven optimal: {proven_milp}/{2 * instances} instances (100 separable + 100 interacting).",
            f"- Same optimal value as enumeration (within {SAME_MILP:g}): {match}/{2 * instances}.", ""]
    tim += [f"## 3. MILP at n = {n}", "",
            f"{sum(milp_ms) / len(milp_ms):.1f} ms per instance on average.", ""]

    # 4. Larger sizes
    for part, (n, k, instances) in zip("ab", LARGE):
        log(f"Part 4{part}: n = {n}, K = {k}")

        def milp_ref(a, b, k=k):
            sel, proven = milp_optimum(a, b, k, MILP_TIME_LIMIT)
            return (value(sel, a, b) if sel is not None else float("-inf")), proven

        stats, proven, ref_ms = evaluate(n, k, instances, milp_ref, SAME_MILP)
        res += [f"## 4{part}. n = {n}, K = {k}, interacting regime, {instances} instances (seeds "
                f"0..{instances - 1})", "",
                f"Reference: MILP optimum, time limit {MILP_TIME_LIMIT:g} s per instance. "
                f"Proven optimal: **{proven}/{instances}**. *Optimal* counts only proven instances "
                f"(same value within {SAME_MILP:g}).", ""]
        res += quality_table(stats, proven)
        res += ["", f"Runs above the proven optimum (must be 0): "
                f"{sum(st.above_reference for st in stats)}."]
        if proven < instances:
            res += ["", f"Unproven instances ({instances - proven}): runs that matched or beat the "
                    "MILP incumbent (not counted as optimal):", ""]
            res += [f"- {st.label}: {st.unproven_match}/{st.unproven_runs}" for st in stats]
        res.append("")
        tim += [f"## 4{part}. n = {n}, K = {k}", "",
                f"MILP: {sum(ref_ms) / len(ref_ms) / 1000:.2f} s per instance on average, "
                f"{max(ref_ms) / 1000:.2f} s at most.", ""]
        tim += timing_table(stats) + [""]

    tim.append(f"Whole script: {time.perf_counter() - t_start:.0f} s.")
    (HERE / "RESULTS_V3.md").write_text("\n".join(res).rstrip() + "\n", encoding="utf-8")
    (HERE / "TIMINGS_V3.md").write_text("\n".join(tim).rstrip() + "\n", encoding="utf-8")
    print("\n".join(res))
    print("\n".join(tim))


if __name__ == "__main__":
    main()
