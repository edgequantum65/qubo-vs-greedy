# Timings (v1/v2 table)

Mean wall-clock time per instance, in milliseconds. Indicative only: measured on Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, Linux, Python 3.11.17, numpy 2.4.1, one process. These numbers change from run to run and machine to machine, so they are kept out of RESULTS.md and are not checked by CI.

| Regime | Method | Mean ms per instance |
|---|---|---:|
| separable (no overlap) | Exact enumeration (C(20,5) = 15 504 subsets) | 2.9 |
| separable (no overlap) | Greedy | 0.1 |
| separable (no overlap) | QUBO + bit-flip annealing (v1), 40 000 flips | 62.4 |
| separable (no overlap) | Swap annealing (v2), 4 000 swaps | 14.4 |
| interacting (30% pairs overlap, up to 15) | Exact enumeration (C(20,5) = 15 504 subsets) | 2.8 |
| interacting (30% pairs overlap, up to 15) | Greedy | 0.1 |
| interacting (30% pairs overlap, up to 15) | QUBO + bit-flip annealing (v1), 40 000 flips | 58.5 |
| interacting (30% pairs overlap, up to 15) | Swap annealing (v2), 4 000 swaps | 10.2 |

Whole script: 15.2 s.
