# Results

n = 20, K = 5, 100 instances per regime, seeds 0..99

Optimal = same value as exact enumeration. Gap = mean shortfall vs the optimum.

| Regime | Method | Optimal | Mean gap |
|---|---|---:|---:|
| separable (no overlap) | Greedy | 100/100 | 0.00% |
| separable (no overlap) | QUBO + bit-flip annealing (v1) | 97/100 | 0.03% |
| separable (no overlap) | Swap annealing, not a QUBO (v2) | 100/100 | 0.00% |
| interacting (30% pairs overlap, up to 15) | Greedy | 74/100 | 1.04% |
| interacting (30% pairs overlap, up to 15) | QUBO + bit-flip annealing (v1) | 8/100 | 5.23% |
| interacting (30% pairs overlap, up to 15) | Swap annealing, not a QUBO (v2) | 98/100 | 0.02% |

**Unequal budgets.** Budget = proposals evaluated per run. The QUBO bit-flip annealer gets 2000 sweeps x 20 = 40 000 proposed flips; the swap annealer gets 200 sweeps x 20 = 4 000 proposed swaps, 10 times fewer. The gap favours the QUBO annealer, which still loses. An equal-budget comparison, with 5 seeds per instance and a greedy + local search baseline, is in RESULTS_V3.md. Timings: TIMINGS.md.
