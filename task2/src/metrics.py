from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        tp = float(np.logical_and(y_true == cls, y_pred == cls).sum())
        fp = float(np.logical_and(y_true != cls, y_pred == cls).sum())
        fn = float(np.logical_and(y_true == cls, y_pred != cls).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(pred)] += 1
    return matrix


def top_confused_pairs(
    matrix: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 10,
) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    for truth_idx in range(matrix.shape[0]):
        for pred_idx in range(matrix.shape[1]):
            if truth_idx == pred_idx:
                continue
            count = int(matrix[truth_idx, pred_idx])
            if count <= 0:
                continue
            rows.append((class_names[truth_idx], class_names[pred_idx], count))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows[:top_k]

