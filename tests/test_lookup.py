import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


# The repository is being migrated to a src layout while the preprocessing
# package still lives at the project root.  Load this module directly so this
# test does not change which package the other tests import.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "traffic_detection"
    / "lookup.py"
)
SPEC = importlib.util.spec_from_file_location("traffic_detection_lookup", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LOOKUP_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOOKUP_MODULE
SPEC.loader.exec_module(LOOKUP_MODULE)
PrefixHashLookup = LOOKUP_MODULE.PrefixHashLookup


class PrefixHashLookupTests(unittest.TestCase):
    def test_support_and_purity_filtering(self):
        repeated = [10, -20, 30, 0]
        unsupported = [99, -1, 0, 0]
        X = np.asarray(
            [repeated, repeated, repeated, repeated, unsupported], dtype=np.int16
        )
        y = np.asarray([7, 7, 7, 9, 9], dtype=np.uint16)

        accepted = PrefixHashLookup(
            [4], min_support=3, min_purity=0.75, hash_batch_size=2
        ).fit(X, y)
        prediction, prefix = accepted.predict(
            np.asarray([repeated, unsupported], dtype=np.int16), return_prefix=True
        )
        np.testing.assert_array_equal(prediction, [7, -1])
        np.testing.assert_array_equal(prefix, [4, 0])
        self.assertEqual(accepted.tables_[0].support.tolist(), [4])
        self.assertEqual(accepted.tables_[0].matched_count.tolist(), [3])

        rejected = PrefixHashLookup(
            [4], min_support=3, min_purity=0.76
        ).fit(X, y)
        np.testing.assert_array_equal(rejected.predict(X[:1]), [-1])

    def test_longest_prefix_backoff(self):
        exact = [10, 20, 30, 40]
        same_prefix = [10, 20, 31, 41]
        X = np.asarray([exact, exact, same_prefix], dtype=np.int16)
        y = np.asarray([101, 101, 202], dtype=np.int16)
        lookup = PrefixHashLookup(
            prefix_lengths=[2, 4, 2],
            min_support=2,
            min_purity=0.6,
            hash_batch_size=1,
        ).fit(X, y)

        query = np.asarray(
            [exact, [10, 20, 99, 99], [80, 20, 30, 40]], dtype=np.int16
        )
        prediction, prefix = lookup.predict(query, return_prefix=True)
        np.testing.assert_array_equal(prediction, [101, 101, -1])
        np.testing.assert_array_equal(prefix, [4, 2, 0])
        self.assertEqual(lookup.prefix_lengths, (4, 2))

    def test_non_contiguous_labels_and_deterministic_tie_break(self):
        row = [1, -2, 0]
        X = np.asarray([row, row], dtype=np.int16)
        y = np.asarray([500, 20], dtype=np.uint16)
        lookup = PrefixHashLookup([3], min_support=2, min_purity=0.5).fit(X, y)
        prediction = lookup.predict(np.asarray([row], dtype=np.int16))
        np.testing.assert_array_equal(prediction, [20])
        np.testing.assert_array_equal(lookup.classes_, [20, 500])

    def test_tables_are_compact_sorted_arrays(self):
        X = np.asarray(
            [[index, -index, 0, 0] for index in range(1, 21)], dtype=np.int16
        )
        y = np.arange(20, dtype=np.uint8)
        lookup = PrefixHashLookup([4], min_support=1).fit(X, y)
        table = lookup.tables_[0]

        self.assertIsInstance(table.keys, np.ndarray)
        self.assertEqual(table.keys.dtype.itemsize, 16)
        self.assertEqual(table.labels.dtype, np.uint8)
        self.assertFalse(any(isinstance(value, dict) for value in vars(lookup).values()))
        self.assertEqual(lookup.nbytes, table.nbytes)
        self.assertLessEqual(table.nbytes, 22 * len(X))

    def test_batch_size_does_not_change_results(self):
        rng = np.random.default_rng(12)
        X = rng.integers(-1448, 1449, size=(200, 30), dtype=np.int16)
        X = np.repeat(X, 2, axis=0)
        y = np.repeat(np.arange(200, dtype=np.uint16), 2)
        query = np.vstack([X[::19], np.zeros((1, 30), dtype=np.int16)])

        small_batches = PrefixHashLookup(
            [30, 12, 5], min_support=2, hash_batch_size=7
        ).fit(X, y)
        large_batches = PrefixHashLookup(
            [30, 12, 5], min_support=2, hash_batch_size=10_000
        ).fit(X, y)
        np.testing.assert_array_equal(
            small_batches.predict(query), large_batches.predict(query)
        )

    def test_validates_input(self):
        lookup = PrefixHashLookup([3])
        with self.assertRaises(TypeError):
            lookup.fit(np.zeros((2, 3), dtype=np.float32), np.asarray([0, 1]))
        with self.assertRaises(ValueError):
            lookup.fit(np.zeros((2, 2), dtype=np.int16), np.asarray([0, 1]))
        with self.assertRaises(RuntimeError):
            lookup.predict(np.zeros((1, 3), dtype=np.int16))


if __name__ == "__main__":
    unittest.main()
