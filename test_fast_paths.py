"""The fast code must give exactly what the v1/v2 code gave, and the MILP must
agree with exact enumeration.

The *_v1v2 functions below are the published v1/v2 implementations, copied
unchanged, used only as references here.
"""
import itertools
import math
import random
import unittest

import numpy as np

from qubo_vs_greedy import (anneal, anneal_swap, exact, local_search, make_instance,
                            milp_optimum, swap_delta, value)


def exact_v1v2(a, b, k):
    best_sel, best_v = None, -math.inf
    for sel in itertools.combinations(range(len(a)), k):
        v = value(sel, a, b)
        if v > best_v + 1e-12:
            best_sel, best_v = sel, v
    return best_sel


def anneal_v1v2(a, b, k, sweeps=2000, t1=0.01, seed=0):
    n = len(a)
    rnd = random.Random(seed)
    penalty = float(max(a.max(), b.sum(axis=1).max()) + 1.0)
    t0 = 2.0 * penalty
    x = [0] * n
    for i in rnd.sample(range(n), k):
        x[i] = 1
    bx = b @ np.array(x, dtype=float)
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


def anneal_swap_v1v2(a, b, k, sweeps=2000, t1=0.01, seed=0):
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


REGIMES = ((0.0, 0.0), (0.3, 15.0))
# Short runs end on different selections for different random streams, so they
# catch any change in the order of random draws; long runs cover the cold end.
SWEEPS = (2, 150)


class FastPathsMatchV1V2(unittest.TestCase):
    def test_exact(self):
        for density, overlap in REGIMES:
            for seed in range(10):
                a, b = make_instance(14, seed, density, overlap)
                self.assertEqual(exact(a, b, 5), exact_v1v2(a, b, 5), (density, seed))

    def test_anneal(self):
        for density, overlap in REGIMES:
            for seed, sweeps in itertools.product(range(10), SWEEPS):
                a, b = make_instance(14, seed, density, overlap)
                self.assertEqual(anneal(a, b, 4, sweeps=sweeps, seed=seed),
                                 anneal_v1v2(a, b, 4, sweeps=sweeps, seed=seed), (density, seed, sweeps))

    def test_anneal_swap(self):
        for density, overlap in REGIMES:
            for seed, sweeps in itertools.product(range(10), SWEEPS):
                a, b = make_instance(14, seed, density, overlap)
                self.assertEqual(anneal_swap(a, b, 4, sweeps=sweeps, seed=seed),
                                 anneal_swap_v1v2(a, b, 4, sweeps=sweeps, seed=seed), (density, seed, sweeps))


class IncrementalDelta(unittest.TestCase):
    def test_swap_delta_matches_value_on_random_swaps(self):
        rnd = random.Random(7)
        for seed in range(5):
            a, b = make_instance(30, seed, 0.3, 15.0)
            al, rows = a.tolist(), b.tolist()
            sel = set(rnd.sample(range(30), 8))
            bx = [sum(rows[j][i] for i in sel) for j in range(30)]
            for _ in range(300):
                out_i = rnd.choice(sorted(sel))
                in_i = rnd.choice([j for j in range(30) if j not in sel])
                new = (sel - {out_i}) | {in_i}
                d = swap_delta(al, rows, bx, out_i, in_i)
                self.assertAlmostEqual(d, value(new, a, b) - value(sel, a, b), delta=1e-9)
                # same incremental update as anneal_swap
                bx = [u + p - q for u, p, q in zip(bx, rows[in_i], rows[out_i])]
                sel = new

    def test_local_search_ends_at_a_swap_local_optimum(self):
        a, b = make_instance(25, 3, 0.3, 15.0)
        sel, _ = local_search(a, b, (0, 1, 2, 3, 4, 5))
        v = value(sel, a, b)
        for out_i in sel:
            for in_i in set(range(25)) - set(sel):
                self.assertLessEqual(value((set(sel) - {out_i}) | {in_i}, a, b), v + 1e-9)


class MilpMatchesEnumeration(unittest.TestCase):
    def test_small_instances(self):
        for density, overlap in REGIMES:
            for seed in range(8):
                a, b = make_instance(12, seed, density, overlap)
                sel, proven = milp_optimum(a, b, 4, time_limit=30.0)
                self.assertTrue(proven)
                self.assertEqual(len(sel), 4)
                self.assertAlmostEqual(value(sel, a, b), value(exact(a, b, 4), a, b), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
