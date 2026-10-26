"""
Training Pipeline for Credit Scoring Model - CodeAlpha Internship Task 1.
Trains and compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
Selects the best performer, computes metrics, and serializes artifacts.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.preprocessing import prepare_train_test_data, CreditDataPreprocessor
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_feature_importance,
    plot_model_comparison,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def get_candidate_models():
    """
    Returns a dictionary of classification models required by the CodeAlpha task.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_split=6,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        ),
    }

    # Optionally attempt to include XGBoost if installed
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            eval_metric="mlogloss",
            random_state=42
        )
    except ImportError:
        pass

    return models


def run_training_pipeline(csv_path: Path = DATA_PATH):
    print("=" * 70)
    print("CodeAlpha Machine Learning Internship - Task 1: Credit Scoring Model")
    print("=" * 70)

    if not csv_path.exists():
        print(f"[ERROR] Data file not found at: {csv_path}")
        print("[INFO] Attempting to download or generate dataset first...")
        from src.download_data import download_from_kagglehub
        download_from_kagglehub()

    print(f"\n[1/5] Loading and Preprocessing Data from: {csv_path.name}")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(csv_path)
    print(f"      Training samples  : {X_train.shape[0]} records, {X_train.shape[1]} features")
    print(f"      Testing samples   : {X_test.shape[0]} records")

    # Save preprocessor artifact
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    models = get_candidate_models()
    results = []
    trained_models = {}

    print("\n[2/5] Training Candidate Models...")
    for name, model in models.items():
        print(f"\n      --> Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = calculate_metrics(model, X_test, y_test)
        results.append({
            "Model": name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision (Macro)": round(metrics["Precision (Macro)"], 4),
            "Recall (Macro)": round(metrics["Recall (Macro)"], 4),
            "F1-Score (Macro)": round(metrics["F1-Score (Macro)"], 4),
            "ROC-AUC (Macro)": round(metrics["ROC-AUC (Macro)"], 4) if metrics["ROC-AUC (Macro)"] else 0.0,
        })

    # Create Summary DataFrame
    results_df = pd.DataFrame(results).sort_values(by="F1-Score (Macro)", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = REPORTS_DIR / "model_benchmark_results.csv"
    results_df.to_csv(results_csv, index=False)

    print("\n[3/5] Benchmark Performance Summary:")
    print("-" * 75)
    print(results_df.to_string(index=False))
    print("-" * 75)
    print(f"Benchmark table saved to: {results_csv}")

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_f1 = results_df.iloc[0]["F1-Score (Macro)"]
    print(f"\n[4/5] Best Model Selected: '{best_model_name}' (Macro F1 = {best_f1:.4f})")

    # Save Best Model
    best_model_path = MODELS_DIR / "best_credit_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"      Saved best model artifact to: {best_model_path}")

    print("\n[5/5] Generating Visual Evaluation Figures...")
    best_metrics = calculate_metrics(best_model, X_test, y_test)
    print("\nDetailed Classification Report for Best Model:")
    print(best_metrics["Classification Report"])

    # Visualizations
    plot_model_comparison(results_df)
    plot_confusion_matrix(best_metrics["Confusion Matrix"], model_name=best_model_name)
    plot_multiclass_roc(best_model, X_test, y_test, model_name=best_model_name)

    # Feature Importance for tree-based best model
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, preprocessor.feature_names, model_name=best_model_name)
    elif "Random Forest" in trained_models and hasattr(trained_models["Random Forest"], "feature_importances_"):
        plot_feature_importance(
            trained_models["Random Forest"], preprocessor.feature_names, model_name="Random Forest"
        )

    print("\n" + "=" * 70)
    print("Training Pipeline Completed Successfully!")
    print(f"Artifacts ready in: {MODELS_DIR}")
    print(f"Evaluation figures ready in: {PROJECT_ROOT / 'reports' / 'figures'}")
    print("=" * 70)

    return best_model, preprocessor, results_df


if __name__ == "__main__":
    run_training_pipeline()
