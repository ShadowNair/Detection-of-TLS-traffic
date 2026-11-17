import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from traffic_detection.data import convert_csv_to_memmap, load_conversion
from traffic_detection.features import (
    FEATURE_NAMES,
    build_feature_memmap,
    extract_features,
    merge_large_same_direction,
)


PACKET_COLUMNS = [f"tcp_len_{index}" for index in range(1, 31)]


class ConversionTests(unittest.TestCase):
    def _write_csv(self, path, first_column, rows):
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow([first_column, *PACKET_COLUMNS])
            writer.writerows(rows)

    def test_train_conversion_preserves_mixed_labels_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.csv"
            self._write_csv(
                source,
                "app_service",
                [
                    ["1", "1448", "-31.0", "null", *(["0.0"] * 27)],
                    ["Telegram", "64.0", "", *(["0"] * 28)],
                    ["1", "-100", *(["0"] * 29)],
                ],
            )
            result = convert_csv_to_memmap(source, root / "converted", split="train")
            packets, labels, classes = load_conversion(result.output_dir, "train")
            self.assertEqual(packets.dtype, np.int16)
            np.testing.assert_array_equal(labels, [0, 1, 0])
            self.assertEqual(classes, ["1", "Telegram"])
            np.testing.assert_array_equal(packets[0, :4], [1448, -31, 0, 0])

    def test_test_conversion_keeps_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "test.csv"
            self._write_csv(source, "id", [["9876543210", *(["0"] * 30)]])
            result = convert_csv_to_memmap(source, root / "converted", split="test")
            packets, ids, classes = load_conversion(result.output_dir, "test")
            self.assertEqual(classes, None)
            self.assertEqual(int(ids[0]), 9_876_543_210)
            self.assertEqual(packets.shape, (1, 30))


class FeatureTests(unittest.TestCase):
    def test_engineered_feature_values(self):
        sequences = np.zeros((2, 30), dtype=np.int16)
        sequences[0, :5] = [100, -200, -300, 1400, 1300]
        features = extract_features(sequences, include_raw=False)
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        row = dict(zip(FEATURE_NAMES, features[0]))
        self.assertEqual(row["flow_length"], 5)
        self.assertEqual(row["client_packets"], 3)
        self.assertEqual(row["server_packets"], 2)
        self.assertEqual(row["client_bytes"], 2800)
        self.assertEqual(row["server_bytes"], 500)
        self.assertEqual(row["direction_changes"], 2)
        self.assertEqual(row["direction_runs"], 3)
        self.assertEqual(row["max_client_run"], 2)
        self.assertEqual(row["max_server_run"], 2)
        self.assertEqual(row["last_signed_size"], 1300)
        self.assertTrue(np.isfinite(features).all())

    def test_large_packet_merge(self):
        sequence = np.zeros((1, 30), dtype=np.int16)
        sequence[0, :8] = [1300, 1400, 100, -1200, -1448, -31, 1250, -1300]
        merged = merge_large_same_direction(sequence)
        np.testing.assert_array_equal(
            merged[0, :7], [2700, 100, -1200, -1448, -31, 1250, -1300]
        )
        self.assertEqual(merged.dtype, np.int32)

    def test_batched_feature_memmap_matches_direct_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_path = root / "packets.npy"
            packet_map = np.lib.format.open_memmap(
                packet_path, mode="w+", dtype=np.int16, shape=(3, 30)
            )
            packet_map[:] = 0
            packet_map[0, :3] = [10, -20, 30]
            packet_map[1, :2] = [1400, 1400]
            packet_map.flush()
            expected = extract_features(np.asarray(packet_map), include_raw=True)
            del packet_map

            output_path = root / "features.npy"
            build_feature_memmap(packet_path, output_path, batch_size=2)
            actual = np.load(output_path, mmap_mode="r")
            np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
