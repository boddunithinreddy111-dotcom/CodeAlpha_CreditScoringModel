"""
Inference and Credit Risk Assessment Module.
Loads the trained credit model and preprocessor to score new applicant profiles.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import CreditDataPreprocessor, LABEL_MAPPING_REV

MODEL_PATH = PROJECT_ROOT / "models" / "best_credit_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"


def load_artifacts():
    """
    Loads saved model and preprocessor pipelines.
    """
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Model or preprocessor missing. Run 'python src/train.py' first.\n"
            f"Expected: {MODEL_PATH} and {PREPROCESSOR_PATH}"
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def get_risk_recommendation(score_class: str, prob_good: float, prob_poor: float) -> str:
    """
    Provides business and financial decision recommendations based on model scores.
    """
    if score_class == "Good":
        return (
            "LOW RISK: Applicant demonstrates high financial reliability, low debt-to-income, "
            "and timely payment history. Recommended for instant loan approval, lower interest rates, "
            "and credit limit enhancements."
        )
    elif score_class == "Standard":
        return (
            "MODERATE RISK: Applicant demonstrates acceptable financial behavior with minor delays "
            "or moderate debt. Recommended for standard rates with routine credit monitoring."
        )
    else:
        return (
            "HIGH RISK: Applicant exhibits elevated debt ratio, frequent payment delays, or excessive "
            "credit inquiries. Recommended to require a co-signer, additional collateral, or offer secured credit products."
        )


def predict_single_applicant(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts creditworthiness class and probability distribution for a single applicant dictionary.
    """
    model, preprocessor = load_artifacts()

    df = pd.DataFrame([applicant_data])
    X_processed = preprocessor.transform(df)

    pred_idx = model.predict(X_processed)[0]
    predicted_label = LABEL_MAPPING_REV[pred_idx]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_processed)[0]
        probabilities = {
            LABEL_MAPPING_REV[i]: round(float(probs[i]), 4) for i in range(len(probs))
        }

    recommendation = get_risk_recommendation(
        predicted_label,
        probabilities.get("Good", 0.0),
        probabilities.get("Poor", 0.0)
    )

    return {
        "Predicted_Score": predicted_label,
        "Class_Index": int(pred_idx),
        "Probabilities": probabilities,
        "Recommendation": recommendation,
    }


def predict_batch_csv(csv_path: Union[str, Path], output_csv: Union[str, Path] = None) -> pd.DataFrame:
    """
    Batch scores a CSV file of applicants and appends credit score predictions.
    """
    model, preprocessor = load_artifacts()
    df_raw = pd.read_csv(csv_path)

    X_processed = preprocessor.transform(df_raw)
    preds = model.predict(X_processed)

    df_out = df_raw.copy()
    df_out["Predicted_Credit_Score"] = [LABEL_MAPPING_REV[idx] for idx in preds]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_processed)
        df_out["Prob_Poor"] = np.round(probs[:, 0], 4)
        df_out["Prob_Standard"] = np.round(probs[:, 1], 4)
        df_out["Prob_Good"] = np.round(probs[:, 2], 4)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        print(f"[SAVED] Predictions written to: {output_csv}")

    return df_out


if __name__ == "__main__":
    # Demo applicant profile
    sample_applicant = {
        "Age": 32,
        "Occupation": "Engineer",
        "Annual_Income": 85000.0,
        "Monthly_Inhand_Salary": 5800.0,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 3,
        "Interest_Rate": 7,
        "Num_of_Loan": 1,
        "Delay_from_due_date": 2,
        "Num_of_Delayed_Payment": 1,
        "Changed_Credit_Limit": 8.5,
        "Num_Credit_Inquiries": 2,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 450.0,
        "Credit_Utilization_Ratio": 22.5,
        "Credit_History_Age": "12 Years and 4 Months",
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 120.0,
        "Amount_invested_monthly": 450.0,
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Monthly_Balance": 4100.0,
    }

    try:
        print("[DEMO] Scoring sample applicant:")
        result = predict_single_applicant(sample_applicant)
        print(f"Predicted Credit Score : {result['Predicted_Score']}")
        print(f"Probabilities          : {result['Probabilities']}")
        print(f"Decision Advice        : {result['Recommendation']}")
    except FileNotFoundError as ex:
        print(str(ex))
