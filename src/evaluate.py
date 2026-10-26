"""
Evaluation and Metrics Visualization Module for Credit Scoring Models.
Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Generates publication-quality charts for internship reports and presentations.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
TARGET_NAMES = ["Poor", "Standard", "Good"]


def calculate_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics as required by the CodeAlpha task.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = np.nan

    metrics = {
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Precision (Weighted)": prec_weighted,
        "Recall (Macro)": rec_macro,
        "Recall (Weighted)": rec_weighted,
        "F1-Score (Macro)": f1_macro,
        "F1-Score (Weighted)": f1_weighted,
        "ROC-AUC (Macro)": roc_auc,
        "Classification Report": classification_report(
            y_test, y_pred, target_names=TARGET_NAMES, zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path = FIGURES_DIR):
    """
    Plots and saves normalized confusion matrix heatmap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        cbar=True,
        square=True
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual True Class", fontsize=12)
    plt.tight_layout()

    filename = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_multiclass_roc(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str, output_dir: Path = FIGURES_DIR):
    """
    Plots One-vs-Rest ROC curve for each credit class.
    """
    if not hasattr(model, "predict_proba"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    y_prob = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"Class '{TARGET_NAMES[i]}' (AUC = {roc_auc_val:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"Multi-Class ROC Curves - {model_name}", fontsize=14, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = output_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_feature_importance(model, feature_names: List[str], model_name: str, top_n: int = 15, output_dir: Path = FIGURES_DIR):
    """
    Plots top-N most influential features for tree-based estimators.
    """
    if not hasattr(model, "feature_importances_"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_scores = [importances[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_features, palette="viridis")
    plt.title(f"Top {top_n} Important Features - {model_name}", fontsize=14, pad=12)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Financial Feature", fontsize=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_model_comparison(results_df, output_dir: Path = FIGURES_DIR):
    """
    Creates a bar chart comparing performance metrics across all models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["Accuracy", "F1-Score (Macro)", "ROC-AUC (Macro)"]
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

    df_melted = results_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="mako")
    plt.ylim(0, 1.05)
    plt.title("Model Benchmark Comparison - Credit Scoring", fontsize=14, pad=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = output_dir / "model_comparison_benchmark.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")
