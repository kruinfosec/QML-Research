"""
Visualization utilities for the MITM Research Implementation Platform.
Generates publication-ready plots: confusion matrices, ROC curves, training curves, SHAP.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          labels=None, save_path=None):
    """Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Plot title.
        labels: Class labels.
        save_path: Path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels or ["Normal", "Attack"],
                yticklabels=labels or ["Normal", "Attack"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    
    plt.show()
    return fig


def plot_roc_curves(results_dict, save_path=None):
    """Plot overlaid ROC curves for all models.
    
    Args:
        results_dict: Dict of {model_name: {'y_true': ..., 'y_proba': ...}}.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (name, data) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        roc_auc = auc(fpr, tpr)
        color = COLORS[i % len(COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")
    
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    
    plt.show()
    return fig


def plot_training_history(history, model_name="Model", save_path=None):
    """Plot training/validation loss and accuracy curves.
    
    Args:
        history: Keras history object or dict with 'loss', 'val_loss', etc.
        model_name: Name for the plot title.
        save_path: Path to save the figure.
    """
    hist = history.history if hasattr(history, "history") else history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(hist["loss"], label="Train Loss", color=COLORS[0], lw=2)
    if "val_loss" in hist:
        ax1.plot(hist["val_loss"], label="Val Loss", color=COLORS[3], lw=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    
    # Accuracy
    ax2.plot(hist["accuracy"], label="Train Accuracy", color=COLORS[1], lw=2)
    if "val_accuracy" in hist:
        ax2.plot(hist["val_accuracy"], label="Val Accuracy", color=COLORS[4], lw=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    
    plt.show()
    return fig


def plot_comparison_bar(metrics_df, metric_cols=None, save_path=None):
    """Plot grouped bar chart comparing models across metrics.
    
    Args:
        metrics_df: DataFrame with model names as index, metrics as columns.
        metric_cols: List of metric columns to plot.
        save_path: Path to save the figure.
    """
    if metric_cols is None:
        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    # Convert percentage strings to floats
    plot_df = metrics_df[metric_cols].copy()
    for col in plot_df.columns:
        plot_df[col] = plot_df[col].astype(str).str.replace("%", "").astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", ax=ax, color=COLORS[:len(metric_cols)], width=0.7)
    
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Model Comparison — Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim([min(plot_df.min().min() - 2, 90), 101])
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    
    plt.show()
    return fig


def plot_shap_summary(model, X, feature_names=None, save_path=None):
    """Generate SHAP feature importance beeswarm plot.
    
    Paper 1: "SHAP-based feature importance analysis reveals that packet size,
    connection duration, and protocol type are three key features."
    
    Args:
        model: Trained Keras model.
        X: Feature data (2D numpy array, small sample recommended).
        feature_names: List of feature names.
        save_path: Path to save the figure.
    """
    import shap
    
    print("[Viz] Computing SHAP values (this may take a few minutes)...")
    
    # Use DeepExplainer for Keras models
    # Take a small background sample
    background = X[:100] if len(X) > 100 else X
    sample = X[:500] if len(X) > 500 else X
    
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(sample)
    except Exception:
        # Fallback to KernelExplainer
        predict_fn = lambda x: model.predict(x, verbose=0)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(sample, nsamples=100)
    
    # Handle shape for SHAP plotting
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) == 3:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)
    
    sample_2d = sample.reshape(sample.shape[0], -1) if len(sample.shape) == 3 else sample
    
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample_2d, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    
    plt.show()
    return fig
