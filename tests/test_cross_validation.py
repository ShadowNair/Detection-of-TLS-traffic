import unittest

import numpy as np

from traffic_detection.cross_validation import balanced_sample_indices, summarize_folds


class CrossValidationTests(unittest.TestCase):
    def test_balanced_sample_caps_each_class_and_is_reproducible(self) -> None:
        labels = np.repeat(np.array([0, 1, 2]), [2, 5, 8])
        first = balanced_sample_indices(labels, cap_per_class=3, seed=17)
        second = balanced_sample_indices(labels, cap_per_class=3, seed=17)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), len(first))
        np.testing.assert_array_equal(np.bincount(labels[first]), [2, 3, 3])

    def test_zero_cap_keeps_all_rows(self) -> None:
        labels = np.array([2, 2, 7, 7])
        np.testing.assert_array_equal(
            balanced_sample_indices(labels, cap_per_class=0, seed=1),
            np.arange(4),
        )

    def test_fold_summary_uses_sample_standard_deviation(self) -> None:
        rows = [
            {"macro_f1": 0.8, "weighted_f1": 0.7, "accuracy": 0.9},
            {"macro_f1": 1.0, "weighted_f1": 0.9, "accuracy": 0.7},
        ]
        summary = summarize_folds(rows)
        self.assertAlmostEqual(summary["macro_f1"]["mean"], 0.9)
        self.assertAlmostEqual(summary["macro_f1"]["std"], np.sqrt(0.02))
        self.assertEqual(summary["accuracy"]["min"], 0.7)


if __name__ == "__main__":
    unittest.main()
