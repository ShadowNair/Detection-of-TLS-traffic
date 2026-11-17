#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CatBoost submission")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    features = np.load(args.data_dir / "test_features.npy", mmap_mode="r")
    ids = np.load(args.data_dir / "test_ids.npy", mmap_mode="r")
    labels = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "app_service"])
        for start in tqdm(range(0, len(features), args.batch_size), desc="prediction"):
            stop = min(start + args.batch_size, len(features))
            pred = np.asarray(model.predict(features[start:stop])).reshape(-1).astype(np.int32)
            writer.writerows(
                (int(row_id), labels[int(label)])
                for row_id, label in zip(ids[start:stop], pred)
            )


if __name__ == "__main__":
    main()

