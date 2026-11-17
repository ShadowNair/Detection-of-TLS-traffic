#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def balanced_sample_indices(
    y: np.ndarray,
    candidates: np.ndarray,
    cap_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_labels = np.asarray(y)[candidates]
    order = np.argsort(candidate_labels, kind="stable")
    sorted_indices = candidates[order]
    sorted_labels = candidate_labels[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(sorted_labels)]
    selected: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group = sorted_indices[start:stop]
        if len(group) > cap_per_class:
            group = rng.choice(group, size=cap_per_class, replace=False)
        selected.append(group)
    result = np.concatenate(selected).astype(np.int32, copy=False)
    rng.shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CatBoost baseline on engineered features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "catboost")
    parser.add_argument("--cap-per-class", type=int, default=2_000)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(args.data_dir / "train_features.npy", mmap_mode="r")
    y = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    all_indices = np.arange(len(y), dtype=np.int32)
    fit_candidates, val_indices = train_test_split(
        all_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=np.asarray(y),
    )
    rng = np.random.default_rng(args.seed)
    fit_indices = balanced_sample_indices(y, fit_candidates, args.cap_per_class, rng)
    del fit_candidates, all_indices

    x_fit = np.asarray(features[fit_indices], dtype=np.float32)
    y_fit = np.asarray(y[fit_indices], dtype=np.int32)
    x_val = np.asarray(features[val_indices], dtype=np.float32)
    y_val = np.asarray(y[val_indices], dtype=np.int32)
    print(f"fit={len(y_fit):,}, validation={len(y_val):,}, features={x_fit.shape[1]}")

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
        task_type=args.task_type,
        border_count=64,
        l2_leaf_reg=4.0,
        random_strength=0.5,
        od_type="Iter",
        od_wait=50,
        verbose=25,
        allow_writing_files=False,
    )
    model.fit(x_fit, y_fit, eval_set=(x_val, y_val), use_best_model=True)
    prediction = np.asarray(model.predict(x_val)).reshape(-1).astype(np.int32)
    metrics = {
        "macro_f1": float(f1_score(y_val, prediction, average="macro")),
        "weighted_f1": float(f1_score(y_val, prediction, average="weighted")),
        "accuracy": float(accuracy_score(y_val, prediction)),
        "fit_rows": int(len(y_fit)),
        "validation_rows": int(len(y_val)),
        "best_iteration": int(model.get_best_iteration()),
    }
    print(json.dumps(metrics, indent=2))
    model.save_model(args.output_dir / "model.cbm")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(args.output_dir / "val_indices.npy", val_indices.astype(np.int32, copy=False))


if __name__ == "__main__":
    main()

