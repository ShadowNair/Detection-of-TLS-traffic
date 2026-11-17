#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.data import convert_csv_to_memmap
from traffic_detection.features import build_feature_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert competition CSV files to NumPy memmaps")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-features", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train = convert_csv_to_memmap(
        args.input_dir / "train.csv",
        args.output_dir,
        split="train",
        overwrite=args.overwrite,
    )
    test = convert_csv_to_memmap(
        args.input_dir / "test.csv",
        args.output_dir,
        split="test",
        overwrite=args.overwrite,
    )
    if args.with_features:
        build_feature_memmap(
            train.packet_path,
            args.output_dir / "train_features.npy",
            batch_size=args.batch_size,
        )
        build_feature_memmap(
            test.packet_path,
            args.output_dir / "test_features.npy",
            batch_size=args.batch_size,
        )
    print(f"Prepared {train.rows:,} train rows and {test.rows:,} test rows in {args.output_dir}")


if __name__ == "__main__":
    main()

