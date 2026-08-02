from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class EvaluationMetrics:
    action_accuracy: float
    message_type_accuracy: float
    overall_accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1_score: dict[str, float]
    confusion_matrix: pd.DataFrame


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def classification_metrics(true_labels: Iterable[str], pred_labels: Iterable[str]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    true_labels = list(true_labels)
    pred_labels = list(pred_labels)
    classes = sorted(set(true_labels) | set(pred_labels))
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    precision = {cls: _safe_divide(tp[cls], tp[cls] + fp[cls]) for cls in classes}
    recall = {cls: _safe_divide(tp[cls], tp[cls] + fn[cls]) for cls in classes}
    f1 = {
        cls: _safe_divide(2 * precision[cls] * recall[cls], precision[cls] + recall[cls])
        for cls in classes
    }
    return precision, recall, f1


def build_confusion_matrix(true_labels: Iterable[str], pred_labels: Iterable[str]) -> pd.DataFrame:
    true_labels = list(true_labels)
    pred_labels = list(pred_labels)
    classes = sorted(set(true_labels) | set(pred_labels))
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for true, pred in zip(true_labels, pred_labels):
        matrix.loc[true, pred] += 1
    matrix.index.name = 'actual'
    matrix.columns.name = 'predicted'
    return matrix
