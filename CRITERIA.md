# Criteria — written 2026-10-28, before the first run

These expectations were written before running `qubo_vs_greedy.py`. They are not changed after seeing the results. Any later change is a new dated version, with this one kept.

- **C1 (separable regime)**: greedy is optimal on 100/100 instances, because with no overlap the best K items are simply the K largest values.
- **C2 (interacting regime)**: greedy is optimal on fewer than 100/100 instances.
- **C3 (interacting regime)**: annealing is optimal on at least as many instances as greedy.
- **Reference cases** (`test_reference_cases.py`): all pass, including the blank control, which must give exactly 0.

All results are published whatever they show, including a failed criterion.

## First run (v1), 2026-10-28

- C1: **passed**, greedy optimal 100/100.
- C2: **passed**, greedy optimal 74/100.
- C3: **failed**, QUBO bit-flip annealing optimal 8/100, against 74/100 for greedy.
- Reference cases: 3/3 passed.

## v2, 2026-10-28, written after seeing the v1 results

The v1 criteria stay as they are, and C3 stays failed for the QUBO annealer.
v2 adds a second annealer (`anneal_swap`) that swaps one item in for one item out, so the "exactly K" constraint never needs a penalty. It is **not a QUBO**. The question for v2: was the failure caused by annealing itself, or by the penalty formulation?

- **C4 (interacting regime)**: swap annealing is optimal on at least as many instances as greedy.

Result: C4 **passed**, swap annealing optimal 98/100 against 74/100 for greedy (mean gap 0.02 % against 1.04 %).

## v3, 2026-10-28, written before the v3 run

The v1 and v2 sections above stay as they are. v3 answers four objections to v1/v2:
the annealers had unequal budgets (QUBO 40 000 proposed flips, swap annealing 4 000
proposed swaps), each ran with one seed, no cheap classical heuristic beyond greedy was
tried, and n = 20 is small enough for exact enumeration.

**Design** (`study_v3.py`, writes RESULTS_V3.md and TIMINGS_V3.md):

- Budget = proposals evaluated per run (one proposed bit flip, or one proposed swap).
  Both annealers run at the same two budgets, 200 n and 2000 n proposals.
- Annealers: 5 runs per instance, seeds instance + 1000 r, r = 0..4. Report the mean
  number of optimal instances over the seeds, the worst and best seed, and a Wilson 95 %
  interval on all runs pooled.
- New classical baseline: greedy followed by first-improvement 1-swap local search
  (deterministic).
- Part 1: n = 20, K = 5, interacting regime, the 100 v1/v2 instances; reference = exact enumeration.
- Part 2: QUBO sensitivity at 2000 n proposals, seed r = 0: penalty x {1, 2, 5} of the
  minimal safe penalty, T0 x {0.1, 0.5, 2} of the penalty.
- Part 3: MILP reference (scipy.optimize.milp / HiGHS; y_ij >= x_i + x_j - 1, y_ij >= 0
  for b_ij > 0, sum x = K) checked against enumeration on the 200 n = 20 instances.
- Part 4: n = 60, K = 10 and n = 100, K = 15, interacting regime (density 0.3, overlap
  up to 15), instances seeds 0..39; reference = MILP optimum with a 120 s time limit per
  instance. "Optimal" is counted only on instances where HiGHS proved optimality; the
  others are reported separately.

**Known before writing this section** (so these are not predictions):
greedy + 1-swap local search is optimal on 86/100 of the n = 20 interacting instances
(measured during review); QUBO annealing with 10 000 proposals, minimal penalty, was
optimal on 4/100, 3/100 and 0/100 with T0 = 2, 0.5 and 0.1 x penalty (measured during
review); exact enumeration, once vectorised, takes about 3 ms per n = 20 instance against
about 10 ms for one swap-annealing run of 4 000 proposals (TIMINGS.md), so at n = 20 no
heuristic is needed; the MILP solve time was probed on n = 60 instances 0-2 (under 1 s)
and n = 100 instances 0-4 (3 to 13 s) to choose the time limit, with no heuristic run on
them. The script was smoke-tested on other instances (seeds 50 000 and up) with tiny budgets.

**Expectations:**

- **C5 (n = 20, equal budgets)**: at both budgets, swap annealing is optimal on at least
  50 more instances than QUBO bit-flip annealing (means over the seeds).
- **C6 (n = 20)**: swap annealing at 4 000 proposals is optimal on at least 95/100 on
  average, above greedy + local search (86/100).
- **C7 (QUBO sensitivity)**: every one of the 9 cells is at most 20/100, so no setting
  comes close to greedy (74/100).
- **C8 (MILP check)**: the MILP optimum equals the enumeration optimum on 200/200 instances.
- **C9 (MILP at scale)**: HiGHS proves optimality on 40/40 instances at n = 60 and on at
  least 36/40 at n = 100 within 120 s each.
- **C10**: greedy's optimal rate is below its n = 20 rate (74 %) at both n = 60 and n = 100.
- **C11**: swap annealing at 2000 n proposals has a mean optimal rate at least that of
  greedy + local search at both sizes, and at least 80 % at n = 60 and 60 % at n = 100.
- **C12**: QUBO bit-flip annealing has a lower optimal rate than greedy at both larger
  sizes and both budgets.

All outcomes will be appended below, passed or failed, without changing this section.

