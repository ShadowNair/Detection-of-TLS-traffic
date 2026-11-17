#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.cross_validation import summarize_folds
from traffic_detection.metrics import confusion_matrix, f1_from_confusion
from traffic_detection.model import PacketCNN


def metric_row(
    target: np.ndarray,
    prediction: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    metrics = f1_from_confusion(confusion_matrix(target, prediction, n_classes))
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "accuracy": float(metrics["accuracy"]),
    }


def cnn_probabilities(
    model: PacketCNN,
    packets: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(packets), model.classifier[-1].out_features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            stop = min(start + batch_size, len(packets))
            batch = torch.as_tensor(
                np.asarray(packets[start:stop], dtype=np.int16).copy(), device=device
            )
            result[start:stop] = torch.softmax(model(batch), dim=1).cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing model weights on stratified parts of their untouched "
            "validation holdout (this is a stability check, not retrained K-fold CV)"
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catboost-model", type=Path)
    parser.add_argument("--ensemble-config", type=Path)
    parser.add_argument("--validation-indices", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--cnn-batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1")
    if args.ensemble_config and not args.catboost_model:
        raise ValueError("--ensemble-config requires --catboost-model")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packets = np.load(args.data_dir / "train_packets.npy", mmap_mode="r")
    labels = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    classes = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))
    n_classes = len(classes)
    if args.validation_indices:
        validation_indices = np.load(args.validation_indices).astype(np.int64, copy=False)
        index_source = str(args.validation_indices)
    else:
        all_indices = np.arange(len(labels), dtype=np.int32)
        _, validation_indices = train_test_split(
            all_indices,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=np.asarray(labels),
        )
        validation_indices = validation_indices.astype(np.int64, copy=False)
        index_source = f"reconstructed: seed={args.seed}, val_size={args.val_size}"
    validation_labels = np.asarray(labels[validation_indices], dtype=np.int32)
    counts = np.bincount(validation_labels, minlength=n_classes)
    if counts.min() < args.folds:
        raise ValueError(
            f"Every class needs at least {args.folds} holdout rows; minimum is {counts.min()}"
        )

    checkpoint = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["num_classes"] != n_classes:
        raise ValueError("Checkpoint and prepared data have different class counts")
    if list(checkpoint["labels"]) != list(classes):
        raise ValueError("Checkpoint and prepared data use different class order")
    cnn_model = PacketCNN(num_classes=n_classes)
    cnn_model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device).eval()

    cat_model: CatBoostClassifier | None = None
    features: np.ndarray | None = None
    if args.catboost_model:
        cat_model = CatBoostClassifier()
        cat_model.load_model(args.catboost_model)
        features = np.load(args.data_dir / "train_features.npy", mmap_mode="r")

    cnn_weight: float | None = None
    if args.ensemble_config:
        ensemble = json.loads(args.ensemble_config.read_text(encoding="utf-8"))
        cnn_weight = float(ensemble["best"]["cnn_weight"])

    predictions = {"cnn": np.empty(len(validation_indices), dtype=np.int16)}
    if cat_model is not None:
        predictions["catboost"] = np.empty(len(validation_indices), dtype=np.int16)
    if cnn_weight is not None:
        predictions["ensemble"] = np.empty(len(validation_indices), dtype=np.int16)

    for start in range(0, len(validation_indices), args.chunk_size):
        stop = min(start + args.chunk_size, len(validation_indices))
        batch_indices = validation_indices[start:stop]
        cnn_probability = cnn_probabilities(
            cnn_model, packets[batch_indices], device, args.cnn_batch_size
        )
        predictions["cnn"][start:stop] = cnn_probability.argmax(axis=1)
        if cat_model is not None and features is not None:
            cat_probability = np.asarray(
                cat_model.predict_proba(features[batch_indices]), dtype=np.float32
            )
            predictions["catboost"][start:stop] = cat_probability.argmax(axis=1)
            if cnn_weight is not None:
                blend = cnn_weight * cnn_probability + (1.0 - cnn_weight) * cat_probability
                predictions["ensemble"][start:stop] = blend.argmax(axis=1)
        print(f"saved-weight evaluation: {stop:,}/{len(validation_indices):,}")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_positions = [
        validation_position
        for _, validation_position in splitter.split(
            np.zeros(len(validation_labels), dtype=np.uint8), validation_labels
        )
    ]
    model_results: dict[str, object] = {}
    for model_name, prediction in predictions.items():
        rows: list[dict[str, float | int]] = []
        for fold, positions in enumerate(fold_positions, start=1):
            rows.append(
                {
                    "fold": fold,
                    "rows": int(len(positions)),
                    **metric_row(validation_labels[positions], prediction[positions], n_classes),
                }
            )
        model_results[model_name] = {
            "holdout_metrics": metric_row(validation_labels, prediction, n_classes),
            "fold_results": rows,
            "fold_summary": summarize_folds(rows),
        }
        np.save(args.output_dir / f"{model_name}_predictions.npy", prediction)

    payload = {
        "evaluation_type": "posthoc_stratified_holdout_parts",
        "warning": (
            "Existing weights are not retrained per fold. Use this to measure score "
            "stability, not as a replacement for K-fold cross-validation."
        ),
        "validation_index_source": index_source,
        "validation_rows": int(len(validation_indices)),
        "folds": args.folds,
        "seed": args.seed,
        "device": str(device),
        "cnn_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "cnn_weight": cnn_weight,
        "models": model_results,
    }
    np.save(args.output_dir / "validation_indices.npy", validation_indices)
    (args.output_dir / "saved_models_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
