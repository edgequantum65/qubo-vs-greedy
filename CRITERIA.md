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

