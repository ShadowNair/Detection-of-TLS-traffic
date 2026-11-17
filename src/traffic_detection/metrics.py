from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def f1_from_confusion(cm: np.ndarray) -> dict[str, float | np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    denom = support + predicted
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    total = support.sum()
    return {
        "macro_f1": float(per_class.mean()),
        "weighted_f1": float(np.dot(per_class, support) / max(total, 1.0)),
        "accuracy": float(tp.sum() / max(total, 1.0)),
        "per_class_f1": per_class,
        "support": support,
    }

