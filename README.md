# qubo-vs-greedy

**Does a "quantum-inspired" formulation beat a plain greedy algorithm?**
A small, reproducible benchmark on one classic problem: pick exactly K items out of n, where items have values and some pairs overlap.

Edge Quantum 65 — R&D lab, Lannemezan, France. Rule of the lab: nothing is published that was not measured.

## Result (n = 20, K = 5, 100 random instances per regime)

| Regime | Method | Optimal | Mean gap |
|---|---|---:|---:|
| separable (no overlap) | Greedy | 100/100 | 0.00% |
| separable (no overlap) | QUBO + bit-flip annealing (v1) | 97/100 | 0.03% |
| separable (no overlap) | Swap annealing, not a QUBO (v2) | 100/100 | 0.00% |
| interacting (30% pairs overlap, up to 15) | Greedy | 74/100 | 1.04% |
| interacting (30% pairs overlap, up to 15) | QUBO + bit-flip annealing (v1) | 8/100 | 5.23% |
| interacting (30% pairs overlap, up to 15) | Swap annealing, not a QUBO (v2) | 98/100 | 0.02% |

"Optimal" = same value as exact enumeration over all C(20,5) = 15 504 subsets.

## What it shows

1. **No interactions: greedy is already optimal.** No annealing, quantum or classical, can do better.
2. **The QUBO form hurt.** Writing "exactly K items" as a penalty term makes single bit flips cross infeasible states. The annealer is optimal 8 times out of 100, far behind greedy.
3. **Annealing itself works.** With moves that keep exactly K items (swap one in, one out), annealing beats greedy: 98/100 against 74/100.

So the gain came from the problem formulation, not from anything quantum. Before claiming a quantum advantage, beat the greedy baseline and the best classical heuristic.

Everything here is classical computation on a laptop. No quantum hardware is used.

## Method

- Criteria were written **before** the first run and are kept unchanged: [CRITERIA.md](CRITERIA.md). Criterion C3 failed and is reported as failed. The swap annealer (v2) was added after seeing the v1 results, and the file says so.
- Reference cases with known answers, including a blank control that must give exactly 0: `test_reference_cases.py`.
- Fixed seeds: every number above is reproducible.

## Run

```bash
pip install numpy
python -m unittest test_reference_cases
python qubo_vs_greedy.py      # about 1 minute, writes RESULTS.md
```

---

**FR** — Une formulation « inspirée du quantique » (QUBO + recuit simulé) bat-elle un simple algorithme glouton ? Sans interactions, le glouton est déjà optimal. Avec interactions, le QUBO à pénalité échoue (8/100) alors qu'un recuit qui respecte la contrainte gagne (98/100 contre 74/100). Le gain vient de la formulation, pas du quantique. Critères écrits avant l'essai, échec compris.

License: MIT.
