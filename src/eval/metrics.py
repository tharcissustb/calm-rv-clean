"""
Evaluation Metrics for CALM-RV
Includes: Macro-F1, Accuracy, ECE, Brier, NLL, and AUPRC
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, log_loss, average_precision_score


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax activation function"""
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins over confidence.
    
    Args:
        probs: [N, C] predicted probabilities
        y_true: [N] true labels
        n_bins: number of confidence bins
    
    Returns:
        ECE value (lower is better)
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """
    Multi-class Brier score.
    
    Args:
        probs: [N, C] predicted probabilities
        y_true: [N] true labels
        n_classes: number of classes
    
    Returns:
        Brier score (lower is better)
    """
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def nll_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Negative Log Likelihood.
    
    Args:
        probs: [N, C] predicted probabilities
        y_true: [N] true labels
    
    Returns:
        NLL value (lower is better)
    """
    eps = 1e-12
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p)))


def auprc_score(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """
    Area Under the Precision-Recall Curve (AUPRC) - macro-averaged.
    AUPRC is more informative than ROC-AUC for imbalanced datasets.
    
    Args:
        probs: [N, C] predicted probabilities
        y_true: [N] true labels
        n_classes: number of classes
    
    Returns:
        Macro-averaged AUPRC (higher is better, range [0, 1])
    """
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    
    auprc_per_class = []
    for c in range(n_classes):
        # Skip classes with no positive samples
        if y_onehot[:, c].sum() > 0:
            ap = average_precision_score(y_onehot[:, c], probs[:, c])
            auprc_per_class.append(ap)
    
    if not auprc_per_class:
        return 0.0
    
    return float(np.mean(auprc_per_class))


def classification_metrics(logits: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    """
    Compute all classification metrics.
    
    Args:
        logits: [N, C] raw logits (pre-softmax)
        y_true: [N] true labels
        n_bins: number of bins for ECE
    
    Returns:
        Dictionary with all metrics
    """
    probs = softmax(logits, axis=1)
    y_pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    ece = ece_score(probs, y_true, n_bins=n_bins)
    brier = brier_score(probs, y_true, n_classes)
    nll = nll_score(probs, y_true)
    auprc = auprc_score(probs, y_true, n_classes)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll),
        "auprc": float(auprc),  # NEW: AUPRC metric
    }