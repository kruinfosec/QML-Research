"""
Evaluation metrics for the MITM Research Implementation Platform.
Computes all metrics from Paper 1: Accuracy, Precision, Recall, F1, FPR, AUC-ROC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute all evaluation metrics.
    
    Paper 1 metrics: Accuracy, Precision, Recall, F1-Score, FPR, AUC-ROC.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for AUC-ROC).
    
    Returns:
        dict: Dictionary of metrics.
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1_score": f1_score(y_true, y_pred, zero_division=0) * 100,
        "fpr": (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0,  # False Positive Rate
        "fnr": (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0.0,  # False Negative Rate
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    
    # AUC-ROC (requires probabilities)
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba) * 100
        except ValueError:
            metrics["auc_roc"] = 0.0
    
    return metrics


def compare_models(results_dict):
    """Create a comparison table from multiple model results.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}.
    
    Returns:
        pd.DataFrame: Comparison table.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        for key in ["accuracy", "precision", "recall", "f1_score", "fpr", "auc_roc"]:
            row[key.replace("_", " ").title()] = f"{metrics.get(key, 0):.2f}%"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("Model")
    
    return df


def print_results(model_name, metrics):
    """Pretty-print metrics for a single model."""
    print(f"\n{'='*50}")
    print(f"  Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"  FPR:       {metrics['fpr']:.2f}%")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.2f}%")
    print(f"  TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print(f"{'='*50}")
