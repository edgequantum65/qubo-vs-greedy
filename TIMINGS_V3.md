# Timings v3

Indicative only: measured on Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, Linux, Python 3.11.17, numpy 2.4.1, one process. Mean wall-clock time per run.

## 1. n = 20, K = 5

Exact enumeration: 2.9 ms per instance.

| Method | Mean ms per run |
|---|---:|
| Greedy | 0.1 |
| Greedy + 1-swap local search | 0.1 |
| QUBO bit-flip annealing, 4 000 flips | 5.7 |
| Swap annealing, 4 000 swaps | 10.1 |
| QUBO bit-flip annealing, 40 000 flips | 55.5 |
| Swap annealing, 40 000 swaps | 98.9 |

## 3. MILP at n = 20

12.7 ms per instance on average.

## 4a. n = 60, K = 10

MILP: 0.59 s per instance on average, 2.00 s at most.

| Method | Mean ms per run |
|---|---:|
| Greedy | 0.3 |
| Greedy + 1-swap local search | 0.8 |
| QUBO bit-flip annealing, 12 000 flips | 16.6 |
| Swap annealing, 12 000 swaps | 35.3 |
| QUBO bit-flip annealing, 120 000 flips | 153.0 |
| Swap annealing, 120 000 swaps | 332.2 |

## 4b. n = 100, K = 15

MILP: 6.30 s per instance on average, 14.80 s at most.

| Method | Mean ms per run |
|---|---:|
| Greedy | 0.7 |
| Greedy + 1-swap local search | 2.5 |
| QUBO bit-flip annealing, 20 000 flips | 29.9 |
| Swap annealing, 20 000 swaps | 68.5 |
| QUBO bit-flip annealing, 200 000 flips | 281.6 |
| Swap annealing, 200 000 swaps | 653.0 |

Whole script: 727 s.
