# qubo-vs-greedy

[![tests](https://github.com/edgequantum65/qubo-vs-greedy/actions/workflows/tests.yml/badge.svg)](https://github.com/edgequantum65/qubo-vs-greedy/actions/workflows/tests.yml)

**Does a "quantum-inspired" formulation beat a plain greedy algorithm?**
A small, reproducible benchmark on one classic problem: pick exactly K items out of n, where items have values and some pairs overlap.

Edge Quantum 65 — R&D lab, Lannemezan, France. Rule of the lab: nothing is published that was not measured.

## Result v1/v2 (n = 20, K = 5, 100 random instances per regime)

| Regime | Method | Optimal | Mean gap |
|---|---|---:|---:|
| separable (no overlap) | Greedy | 100/100 | 0.00% |
| separable (no overlap) | QUBO + bit-flip annealing (v1) | 97/100 | 0.03% |
| separable (no overlap) | Swap annealing, not a QUBO (v2) | 100/100 | 0.00% |
| interacting (30% pairs overlap, up to 15) | Greedy | 74/100 | 1.04% |
| interacting (30% pairs overlap, up to 15) | QUBO + bit-flip annealing (v1) | 8/100 | 5.23% |
| interacting (30% pairs overlap, up to 15) | Swap annealing, not a QUBO (v2) | 98/100 | 0.02% |

"Optimal" = same value as exact enumeration over all C(20,5) = 15 504 subsets.

**Budgets were unequal.** Budget = proposals evaluated per run. The QUBO annealer got 2000 sweeps x 20 = 40 000 proposed bit flips, the swap annealer 200 sweeps x 20 = 4 000 proposed swaps, 10 times fewer. This favoured the QUBO annealer, which still lost. The v3 study below compares them at equal budgets.

Mean time per instance, interacting regime ([TIMINGS.md](TIMINGS.md), indicative): exact enumeration 2.8 ms, greedy 0.1 ms, QUBO annealing 58.5 ms, swap annealing 10.2 ms. At n = 20 the exact answer is the cheapest one.

## v3: equal budgets, 5 seeds, a stronger baseline, larger sizes

Pre-registered in [CRITERIA.md](CRITERIA.md) (section v3, committed before the run): 8 expectations, C5 to C12. **All 8 passed.** Full tables: [RESULTS_V3.md](RESULTS_V3.md); timings: [TIMINGS_V3.md](TIMINGS_V3.md).

Budget = proposals evaluated per run (one proposed bit flip, or one proposed swap); both annealers get the same budgets, 200 n and 2000 n. Annealers: 5 seeds per instance, mean over the seeds (worst-best seed in RESULTS_V3.md). Greedy + local search = greedy, then first-improvement 1-swap hill climbing, deterministic.

**n = 20, K = 5, interacting regime, the same 100 instances.** Reference: exact enumeration.

| Method | Budget (proposals) | Optimal | Mean gap | ms per run |
|---|---:|---:|---:|---:|
| Greedy | – | 74/100 | 1.04% | 0.1 |
| Greedy + 1-swap local search | 85 swaps (mean) | 86/100 | 0.39% | 0.1 |
| QUBO + bit-flip annealing | 4 000 | 1/100 | 16.01% | 5.7 |
| Swap annealing | 4 000 | 99.2/100 | 0.02% | 10.1 |
| QUBO + bit-flip annealing | 40 000 | 12.2/100 | 5.35% | 55.5 |
| Swap annealing | 40 000 | 100/100 | 0.00% | 98.9 |
| Exact enumeration | 15 504 subsets | 100/100 | 0.00% | 2.9 |

**Larger sizes, interacting regime, 40 instances each.** Enumeration is out of reach (C(100,15) is about 2.5 x 10^17), so the reference is an MILP solved by HiGHS (`scipy.optimize.milp`), which proved the optimum on 40/40 instances at both sizes. The MILP was first checked against enumeration: same optimum on all 200 n = 20 instances.

| Method | Budget | n = 60, K = 10: optimal | gap | ms | n = 100, K = 15: optimal | gap | ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy | – | 7/40 | 4.53% | 0.3 | 2/40 | 7.99% | 0.7 |
| Greedy + 1-swap local search | – | 14/40 | 2.39% | 0.8 | 5/40 | 4.67% | 2.5 |
| QUBO + bit-flip annealing | 200 n | 0/40 | 46.91% | 16.6 | 0/40 | 79.63% | 29.9 |
| Swap annealing | 200 n | 22/40 | 0.94% | 35.3 | 14.4/40 | 2.10% | 68.5 |
| QUBO + bit-flip annealing | 2000 n | 0/40 | 33.98% | 153 | 0/40 | 64.36% | 282 |
| Swap annealing | 2000 n | 36.8/40 | 0.04% | 332 | 27.8/40 | 0.44% | 653 |
| MILP (HiGHS), exact | – | 40/40 | 0.00% | 590 | 40/40 | 0.00% | 6 300 |

**QUBO sensitivity** (n = 20, interacting, 40 000 proposals, seed 0): optimal instances out of 100. The v1 failure is not a tuning artefact.

| Penalty (x minimal safe value) | T0 = 0.1 x penalty | T0 = 0.5 x penalty | T0 = 2 x penalty |
|---|---:|---:|---:|
| 1 x | 0 | 9 | 8 (v1 setting) |
| 2 x | 0 | 0 | 6 |
| 5 x | 0 | 2 | 5 |

Timings are indicative only: one process on a cloud container (Intel Xeon, 2.8 GHz). They are in separate files that CI does not check.

## What it shows

1. **No interactions: greedy is already optimal.** No annealing, quantum or classical, can do better.
2. **The QUBO form hurt, at both budgets and with every setting tried.** Writing "exactly K items" as a penalty term makes single bit flips cross infeasible states. At equal budgets the QUBO annealer is optimal on 1/100 and 12.2/100 instances at n = 20 (means over 5 seeds), and 0/40 at n = 60 and n = 100. None of the 9 penalty and temperature settings gets above 9/100.
3. **Annealing itself works.** With moves that keep exactly K items, annealing beats greedy and greedy + local search. The margin grows with size: 92 % against 35 % optimal at n = 60, and 70 % against 12.5 % at n = 100.
4. **At these sizes an exact method is cheap.** At n = 20, enumeration (3 ms) is faster than any annealing run, so no heuristic is needed. At n = 100 the MILP proves the optimum in 6 s on average (15 s at most), while the best heuristic run (0.65 s) still misses it in about 30 % of runs.

So the gain came from the problem formulation, not from anything quantum. Before claiming a quantum advantage, beat greedy, greedy + local search, an annealer whose moves respect the constraints, and an exact solver wherever one finishes in seconds.

Everything here is classical computation on an ordinary CPU. No quantum hardware is used.

## Method

- Criteria were written **before** the first run and are kept unchanged: [CRITERIA.md](CRITERIA.md). Criterion C3 failed and is reported as failed. The swap annealer (v2) was added after seeing the v1 results, and the file says so.
- Reference cases with known answers, including a blank control that must give exactly 0: `test_reference_cases.py`.
- v3 expectations were written and committed **before** the v3 run (see the git history of CRITERIA.md); the outcomes are appended below them.
- Fixed seeds: every quality number above is reproducible. RESULTS_V3.md came out byte-identical on two separate runs; it depends on the MILP proving every optimum within its 120 s time limit, which it did here in at most 15 s.

## Run

```bash
pip install -r requirements.txt                       # numpy, scipy (for the MILP)
python -m unittest test_reference_cases test_fast_paths
python qubo_vs_greedy.py   # v1/v2 table: about 15 s, writes RESULTS.md and TIMINGS.md
python study_v3.py         # v3 study: about 12 min, writes RESULTS_V3.md and TIMINGS_V3.md
```

CI (GitHub Actions) runs the tests, regenerates the v1/v2 table and fails if RESULTS.md changes by a single byte. It does not run the v3 study (too long, and the MILP has a time limit) and does not check timings.

The code was sped up for v3 (whole v1/v2 script: 77 s before, 16 s after, same machine) without changing any result: the annealers score proposals incrementally and enumeration is vectorised, with every near-tie decision re-checked the v1/v2 way. `test_fast_paths.py` checks the new code against the v1/v2 code, and the regenerated v1/v2 table is byte-identical.

---

**FR** — Une formulation « inspirée du quantique » (QUBO + recuit simulé) bat-elle un simple algorithme glouton ? Sans interactions, le glouton est déjà optimal. Avec interactions, le QUBO à pénalité échoue, à budget égal et quels que soient la pénalité et la température testées (au mieux 9/100), alors qu'un recuit qui respecte la contrainte gagne (99/100 contre 74/100 pour le glouton et 86/100 pour glouton + recherche locale ; à n = 100, 70 % contre 12,5 %). Mais à ces tailles une méthode exacte suffit : énumération en 3 ms à n = 20, MILP prouvé optimal en 6 s en moyenne à n = 100. Le gain vient de la formulation, pas du quantique. Critères écrits avant chaque essai, échec compris.

License: MIT.
