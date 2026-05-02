from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    false_accept_rate: float
    false_reject_rate: float
    confusion_matrix: np.ndarray


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    unknown_label: str = "Unknown",
) -> EvaluationReport:
    known_total = sum(label != unknown_label for label in y_true)
    unknown_total = sum(label == unknown_label for label in y_true)
    false_rejects = sum(true != unknown_label and pred == unknown_label for true, pred in zip(y_true, y_pred))
    false_accepts = sum(true == unknown_label and pred != unknown_label for true, pred in zip(y_true, y_pred))
    labels = sorted(set(y_true) | set(y_pred))

    return EvaluationReport(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        false_accept_rate=false_accepts / max(unknown_total, 1),
        false_reject_rate=false_rejects / max(known_total, 1),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
    )
