import unittest

import numpy as np

from scripts.generate_evaluation_diagnostics import expected_calibration_error


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_calibration_error(self) -> None:
        target = np.asarray([0, 1, 2, 1], dtype=np.int32)
        probability = np.eye(3, dtype=np.float32)[target]
        error, confidence, accuracy, counts = expected_calibration_error(
            target, probability, bins=5
        )
        self.assertAlmostEqual(error, 0.0)
        self.assertEqual(int(counts.sum()), len(target))
        np.testing.assert_allclose(confidence[counts > 0], accuracy[counts > 0])

    def test_overconfident_wrong_prediction_increases_error(self) -> None:
        target = np.asarray([0, 1], dtype=np.int32)
        probability = np.asarray([[0.9, 0.1], [0.9, 0.1]], dtype=np.float32)
        error, _, _, _ = expected_calibration_error(target, probability, bins=5)
        self.assertAlmostEqual(error, 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
