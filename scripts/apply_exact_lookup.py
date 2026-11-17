#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.lookup import PrefixHashLookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Override model predictions for exact train/test matches")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args()

    train_x = np.load(args.data_dir / "train_packets.npy", mmap_mode="r")
    train_y = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    test_x = np.load(args.data_dir / "test_packets.npy", mmap_mode="r")
    test_ids = np.load(args.data_dir / "test_ids.npy", mmap_mode="r")
    labels = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))

    # Train rows are unique, so support=1 is intentional for the exact table.
    lookup = PrefixHashLookup(
        prefix_lengths=[30],
        min_support=1,
        min_purity=1.0,
        hash_batch_size=args.batch_size,
    ).fit(train_x, train_y)
    exact_prediction = lookup.predict(test_x, default_label=-1)
    covered = exact_prediction >= 0
    print(f"Exact lookup coverage: {int(covered.sum()):,}/{len(covered):,} ({covered.mean():.4%})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.base_submission.open("r", newline="", encoding="utf-8") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.reader(source)
        writer = csv.writer(target)
        header = next(reader)
        if header != ["id", "app_service"]:
            raise ValueError(f"Unexpected submission header: {header}")
        writer.writerow(header)
        row_count = 0
        for row_count, row in enumerate(tqdm(reader, total=len(test_ids), desc="lookup override"), start=1):
            index = row_count - 1
            row_id = int(row[0])
            if row_id != int(test_ids[index]):
                raise ValueError(f"Submission/test id mismatch at row {row_count}: {row_id} != {test_ids[index]}")
            label = labels[int(exact_prediction[index])] if covered[index] else row[1]
            writer.writerow((row_id, label))
        if row_count != len(test_ids):
            raise ValueError(f"Submission has {row_count} rows, expected {len(test_ids)}")


if __name__ == "__main__":
    main()

