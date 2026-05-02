import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def top1_accuracy(y_true: list, y_pred: list) -> float:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def compute_far_frr(
    y_true: list, y_pred: list, unknown_label: str = "Unknown"
) -> tuple[float, float]:
    """
    FAR = False Accept Rate  (impostor accepted as genuine)
    FRR = False Reject Rate  (genuine rejected as unknown)
    """
    genuine = [(t, p) for t, p in zip(y_true, y_pred) if t != unknown_label]
    impostor = [(t, p) for t, p in zip(y_true, y_pred) if t == unknown_label]

    frr = sum(p == unknown_label for _, p in genuine) / max(len(genuine), 1)
    far = sum(p != unknown_label for _, p in impostor) / max(len(impostor), 1)
    return far, frr


def compute_eer(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> float:
    """Equal Error Rate via ROC interpolation."""
    labels = np.concatenate([np.ones(len(scores_genuine)), np.zeros(len(scores_impostor))])
    scores = np.concatenate([scores_genuine, scores_impostor])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def roc_data(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> tuple:
    labels = np.concatenate([np.ones(len(scores_genuine)), np.zeros(len(scores_impostor))])
    scores = np.concatenate([scores_genuine, scores_impostor])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def get_confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
