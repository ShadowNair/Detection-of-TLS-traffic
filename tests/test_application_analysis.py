import unittest

import numpy as np

from scripts.analyze_application_patterns import build_profile_frame, update_accumulators


class ApplicationAnalysisTests(unittest.TestCase):
    def test_profile_aggregation_respects_padding_sign_and_order(self) -> None:
        packets = np.zeros((3, 30), dtype=np.int16)
        packets[0, :2] = [10, -20]
        packets[1, :3] = [10, 20, -30]
        packets[2, :4] = [1400, -1400, -60, 60]
        labels = np.asarray(["A", "A", "B"])
        accumulators = {}

        # defaultdict behaviour is used by the production collector.
        from collections import defaultdict
        from scripts.analyze_application_patterns import ClassAccumulator

        mutable = defaultdict(ClassAccumulator)
        update_accumulators(mutable, labels, packets)
        profiles, positions = build_profile_frame(dict(mutable))
        profile_a = profiles.set_index("app_service").loc["A"]

        self.assertEqual(profile_a["flows"], 2)
        self.assertEqual(profile_a["packets"], 5)
        self.assertAlmostEqual(profile_a["mean_flow_length"], 2.5)
        self.assertEqual(profile_a["median_flow_length"], 2)
        self.assertAlmostEqual(profile_a["client_packet_share"], 3 / 5)
        self.assertAlmostEqual(profile_a["client_byte_share"], 40 / 90)
        self.assertAlmostEqual(profile_a["mean_abs_packet_size"], 18.0)
        self.assertAlmostEqual(profile_a["mean_direction_changes"], 1.0)
        self.assertAlmostEqual(profile_a["small_packet_share"], 1.0)
        self.assertAlmostEqual(profile_a["large_packet_share"], 0.0)
        self.assertAlmostEqual(positions["A"][0, 2], 0.5)
        self.assertAlmostEqual(positions["A"][1, 2], 0.0)

        profile_b = profiles.set_index("app_service").loc["B"]
        self.assertAlmostEqual(profile_b["large_packet_share"], 0.5)
        self.assertAlmostEqual(profile_b["near_mtu_packet_share"], 0.5)
        self.assertAlmostEqual(profile_b["mean_direction_changes"], 2.0)


if __name__ == "__main__":
    unittest.main()
