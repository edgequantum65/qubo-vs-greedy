"""Reference cases with known answers, including a blank control that must give zero."""
import unittest

import numpy as np

from qubo_vs_greedy import anneal, exact, greedy, value


class ReferenceCases(unittest.TestCase):
    def test_separable_known_optimum(self):
        a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        b = np.zeros((5, 5))
        for method in (exact, greedy, anneal):
            self.assertEqual(value(method(a, b, 2), a, b), 9.0, method.__name__)

    def test_overlap_trap_where_greedy_fails(self):
        # Item 0 is the best alone but overlaps with both others.
        a = np.array([10.0, 9.0, 9.0])
        b = np.zeros((3, 3))
        b[0, 1] = b[1, 0] = 8.0
        b[0, 2] = b[2, 0] = 8.0
        self.assertEqual(value(exact(a, b, 2), a, b), 18.0)
        self.assertEqual(value(greedy(a, b, 2), a, b), 11.0)
        self.assertEqual(value(anneal(a, b, 2), a, b), 18.0)

    def test_blank_control_gives_zero(self):
        a = np.zeros(6)
        b = np.zeros((6, 6))
        for method in (exact, greedy, anneal):
            self.assertEqual(value(method(a, b, 3), a, b), 0.0, method.__name__)


if __name__ == "__main__":
    unittest.main()
