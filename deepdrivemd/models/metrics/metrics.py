"""Functions for computing evaluation metrics in an unsupervised setting."""
import numpy as np
from typing import NamedTuple


class ConfusionMatrix(NamedTuple):
    tp: int
    tn: int
    fp: int
    fn: int


def _validate_shape(x_true: np.ndarray, x_pred: np.ndarray) -> None:
    if x_true.shape != x_pred.shape:
        raise ValueError("Shape mismatch true {x_true.shape} != pred {x_pred.shape}")


def _validate_metrics(metrics: list) -> None:
    valid_metrics = {"accuracy", "precision", "recall"}
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(f"metric {metric} not supported")


def pixel_confusion_matrix(x_true: np.ndarray, x_pred: np.ndarray) -> ConfusionMatrix:
    """Computes tp, tn, fp, fn for a single example"""
    tp, tn, fp, fn = 0, 0, 0, 0
    for xi, xj in zip(x_true.flatten(), x_pred.flatten()):
        # Round to closest int (0 or 1)
        xi, xj = round(xi), round(xj)
        if xi == 1 and xj == 1:
            tp += 1
        elif xi == 0 and xj == 0:
            tn += 1
        elif xi == 0 and xj == 1:
            fp += 1
        elif xi == 1 and xj == 0:
            fn += 1
        else:
            raise ValueError("round failed to yield 0 or 1: true {x_i}, pred {x_j}")
    return ConfusionMatrix(tp, tn, fp, fn)


def precision_score(cm: ConfusionMatrix) -> float:
    try:
        return cm.tp / (cm.tp + cm.fp)
    except ZeroDivisionError:
        return 0.0


def recall_score(cm: ConfusionMatrix) -> float:
    try:
        return cm.tp / (cm.tp + cm.fn)
    except ZeroDivisionError:
        return 0.0


def accuracy_score(cm: ConfusionMatrix) -> float:
    try:
        correct = cm.tp + cm.tn
        incorrect = cm.fp + cm.fn
        return correct / (correct + incorrect)
    except ZeroDivisionError:
        return 0.0


def pixel_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    metrics=["recall", "precision", "accuracy"],
    normalize=True,
) -> dict:
    """
    Parameters
    ----------
    x_true: np.ndarray
        (N, M, M) array of true matrices
    x_pred: np.ndarray
        (N, M, M) array of predicted matrices
    metrics: list
        metrics to compute
    normalize: bool
        If True, return the metric scores averaged over
        all of the input arrays. If False, return the
        scores per example in a list with the same
        ordering as the input arrays.

    Returns
    -------
    dict: dictionary of metrics mapped to list or normalized scores
    """
    # Make sure input arrays have the same shape
    _validate_shape(x_true, x_pred)
    _validate_metrics(metrics)

    scores = {metric: [] for metric in metrics}

    # Function dispatch
    metric_func = {
        "precision": precision_score,
        "recall": recall_score,
        "accuracy": accuracy_score,
    }

    for true, pred in zip(x_true, x_pred):
        confusion_matrix = pixel_confusion_matrix(true, pred)

        for metric in metrics:
            scores[metric].append(metric_func[metric](confusion_matrix))

    if normalize:
        for metric in metrics:
            scores[metric] = np.mean(scores[metric])

    return scores
