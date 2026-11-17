#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.cross_validation import balanced_sample_indices, summarize_folds


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified K-fold cross-validation for the CatBoost model"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cap-per-class", type=int, default=2_000)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(args.data_dir / "train_features.npy", mmap_mode="r")
    y = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    sample_indices = balanced_sample_indices(y, args.cap_per_class, args.seed)
    sample_labels = np.asarray(y[sample_indices], dtype=np.int32)
    n_classes = int(sample_labels.max()) + 1
    counts = np.bincount(sample_labels, minlength=n_classes)
    if counts.min() < args.folds:
        raise ValueError(
            f"Every class needs at least {args.folds} rows; minimum is {counts.min()}"
        )

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    oof_prediction = np.full(len(sample_indices), -1, dtype=np.int16)
    fold_rows: list[dict[str, float | int]] = []

    print(
        f"CatBoost CV: rows={len(sample_indices):,}, classes={n_classes}, "
        f"folds={args.folds}, features={features.shape[1]}"
    )
    for fold, (fit_local, val_local) in enumerate(
        splitter.split(np.zeros(len(sample_labels), dtype=np.uint8), sample_labels),
        start=1,
    ):
        fit_indices = sample_indices[fit_local]
        val_indices = sample_indices[val_local]
        x_fit = np.asarray(features[fit_indices], dtype=np.float32)
        y_fit = sample_labels[fit_local]
        x_val = np.asarray(features[val_indices], dtype=np.float32)
        y_val = sample_labels[val_local]

        print(
            f"fold {fold}/{args.folds}: fit={len(fit_indices):,}, "
            f"validation={len(val_indices):,}"
        )
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            random_seed=args.seed + fold,
            task_type=args.task_type,
            border_count=64,
            l2_leaf_reg=4.0,
            random_strength=0.5,
            od_type="Iter",
            od_wait=args.early_stopping_rounds,
            verbose=max(1, args.iterations // 10),
            allow_writing_files=False,
        )
        model.fit(x_fit, y_fit, eval_set=(x_val, y_val), use_best_model=True)
        prediction = np.asarray(model.predict(x_val)).reshape(-1).astype(np.int16)
        oof_prediction[val_local] = prediction
        row: dict[str, float | int] = {
            "fold": fold,
            "fit_rows": int(len(fit_indices)),
            "validation_rows": int(len(val_indices)),
            "best_iteration": int(model.get_best_iteration()),
            **metric_row(y_val, prediction),
        }
        fold_rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        if args.save_models:
            model.save_model(args.output_dir / f"fold_{fold}.cbm")

        del model, x_fit, y_fit, x_val, y_val, prediction
        gc.collect()

    if np.any(oof_prediction < 0):
        raise RuntimeError("Some rows did not receive an out-of-fold prediction")

    aggregate = metric_row(sample_labels, oof_prediction)
    summary = summarize_folds(fold_rows)
    payload = {
        "model": "CatBoost",
        "folds": args.folds,
        "seed": args.seed,
        "cap_per_class": args.cap_per_class,
        "sample_rows": int(len(sample_indices)),
        "classes": n_classes,
        "fold_results": fold_rows,
        "fold_summary": summary,
        "oof_metrics": aggregate,
    }
    (args.output_dir / "catboost_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(args.output_dir / "sample_indices.npy", sample_indices)
    np.save(args.output_dir / "oof_predictions.npy", oof_prediction)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

