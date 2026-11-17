from __future__ import annotations

import numpy as np


def balanced_sample_indices(
    labels: np.ndarray,
    cap_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return shuffled row indices with at most ``cap_per_class`` per class."""

    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    all_indices = np.arange(len(y), dtype=np.int64)
    if cap_per_class <= 0 or len(y) == 0:
        return all_indices

    order = np.argsort(y, kind="stable")
    sorted_indices = all_indices[order]
    sorted_labels = y[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(y)]
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group = sorted_indices[start:stop]
        if len(group) > cap_per_class:
            group = rng.choice(group, size=cap_per_class, replace=False)
        selected.append(group)
    result = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


def summarize_folds(
    fold_rows: list[dict[str, float | int]],
    metric_names: tuple[str, ...] = ("macro_f1", "weighted_f1", "accuracy"),
) -> dict[str, dict[str, float]]:
    """Calculate mean, sample standard deviation and range across folds."""

    if len(fold_rows) < 2:
        raise ValueError("At least two folds are required")
    summary: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in fold_rows], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary
