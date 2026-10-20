import unittest, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import clf, reg


class AnalysisChecks(unittest.TestCase):
    def test_metrics_and_input_invariants(self):
        m = json.loads((ROOT / "results/metrics.json").read_text())
        self.assertLess(m["scalar_global"]["x"], 0)
        self.assertLess(m["constrained_projection"]["max_violation"], 1e-6)
        self.assertTrue(all(t["objective"] < 1e-8 for t in m["trust_regions"]))

    def test_cg_exact_solution_and_invalid_matrix(self):
        from analysis import conjugate_gradient

        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x, h = conjugate_gradient(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)
        with self.assertRaises(ValueError):
            conjugate_gradient(np.array([[1.0, 2.0], [0.0, 1.0]]), b)

    def test_metric_semantics(self):
        m = clf([0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.8, 0.9, 0.2])
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["confusion"], [[1, 1], [1, 1]])
        self.assertEqual(reg([1, 2, 3], [1, 2, 3])["rmse"], 0)


if __name__ == "__main__":
    unittest.main()
