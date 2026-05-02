import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_roc_curves(
    model_data: dict[str, tuple],   # {model_name: (fpr, tpr, auc)}
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, roc_auc) in model_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_per_student_accuracy(
    student_ids: list[str], accuracies: list[float], save_path: Path | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(student_ids) // 2), 5))
    ax.bar(student_ids, accuracies)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Student ID")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Per-Student Recognition Accuracy")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
