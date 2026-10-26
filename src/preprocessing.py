"""
Preprocessing and Feature Engineering Pipeline for Credit Scoring Model.
Handles raw string sanitation, history duration parsing, ratio engineering,
imputation, encoding, and scaling.
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET_COL = "Credit_Score"
LABEL_MAPPING = {"Poor": 0, "Standard": 1, "Good": 2}
LABEL_MAPPING_REV = {0: "Poor", 1: "Standard", 2: "Good"}

DROP_COLS = ["ID", "Customer_ID", "Month", "Name", "SSN", "Type_of_Loan"]


def parse_credit_history_age(val) -> float:
    """
    Parses strings like '23 Years and 5 Months' into total months (float).
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r"(\d+)\s*Years?\s*(?:and)?\s*(\d+)?\s*Months?", val_str, re.IGNORECASE)
    if match:
        years = float(match.group(1))
        months = float(match.group(2)) if match.group(2) else 0.0
        return years * 12.0 + months
    # Fallback if only digits
    digits = re.findall(r"\d+", val_str)
    if digits:
        return float(digits[0]) * 12.0
    return np.nan


def clean_numeric_col(series: pd.Series) -> pd.Series:
    """
    Strips trailing underscores and non-numeric artifacts from raw data.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning on raw dataframe columns.
    """
    df = df.copy()

    # Drop non-predictive identifiers if present
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Numeric columns that often contain dirty string characters in Kaggle raw data
    dirty_numeric_cols = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Delay_from_due_date",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Monthly_Inhand_Salary", "Credit_Utilization_Ratio", "Total_EMI_per_month"
    ]
    for col in dirty_numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_col(df[col])

    # Parse Credit History Age to total months
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Months"] = df["Credit_History_Age"].apply(parse_credit_history_age)
        df = df.drop(columns=["Credit_History_Age"])

    # Clean Payment_of_Min_Amount
    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount"] = (
            df["Payment_of_Min_Amount"].astype(str).str.strip().map({"Yes": 1, "No": 0}).fillna(0)
        )

    # Clean Credit_Mix (Good: 2, Standard: 1, Bad: 0, missing/invalid: 1)
    if "Credit_Mix" in df.columns:
        df["Credit_Mix_Score"] = (
            df["Credit_Mix"].astype(str).str.strip().map({"Good": 2, "Standard": 1, "Bad": 0}).fillna(1)
        )
        df = df.drop(columns=["Credit_Mix"])

    # Feature Engineering (Key financial indicators)
    income_safe = df["Annual_Income"].clip(lower=1000.0)
    salary_safe = df["Monthly_Inhand_Salary"].clip(lower=100.0)

    # 1. Debt to Annual Income ratio
    df["Debt_to_Income_Ratio"] = df["Outstanding_Debt"] / income_safe
    # 2. Monthly Debt Service (EMI) to Monthly Base Salary ratio
    df["EMI_to_Salary_Ratio"] = df["Total_EMI_per_month"] / salary_safe
    # 3. Monthly Investment to Monthly Base Salary ratio
    df["Investment_to_Salary_Ratio"] = df["Amount_invested_monthly"] / salary_safe
    # 4. Delay Severity indicator (Days delayed * count of delayed payments)
    df["Delay_Severity"] = df["Delay_from_due_date"] * df["Num_of_Delayed_Payment"]
    # 5. Financial Buffer / Net Savings Ratio
    df["Savings_to_Salary_Ratio"] = df["Monthly_Balance"] / salary_safe

    return df


class CreditDataPreprocessor:
    """
    End-to-end Preprocessor that transforms raw credit records into model-ready features.
    """

    def __init__(self):
        self.num_pipeline = None
        self.cat_pipeline = None
        self.full_pipeline = None
        self.num_cols = []
        self.cat_cols = []
        self.feature_names = []

    def fit(self, X: pd.DataFrame):
        X_clean = clean_dataframe(X)

        self.cat_cols = [
            c for c in ["Occupation", "Payment_Behaviour"] if c in X_clean.columns
        ]
        self.num_cols = [
            c for c in X_clean.columns if c not in self.cat_cols and c != TARGET_COL
        ]

        self.num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        self.cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.full_pipeline = ColumnTransformer(
            transformers=[
                ("num", self.num_pipeline, self.num_cols),
                ("cat", self.cat_pipeline, self.cat_cols)
            ]
        )

        self.full_pipeline.fit(X_clean)

        # Cache transformed feature names for explainability & feature importance
        cat_encoder = self.full_pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(self.cat_cols))
        self.feature_names = self.num_cols + cat_feature_names

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = clean_dataframe(X)
        return self.full_pipeline.transform(X_clean)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: str or Path):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[SAVED] Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str or Path):
        return joblib.load(filepath)


def prepare_train_test_data(
    csv_path: str or Path, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CreditDataPreprocessor]:
    """
    Loads raw CSV, prepares features, splits into stratified train/test sets,
    and returns scaled matrices, label vectors, and fitted preprocessor.
    """
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")

    # Drop rows where target is missing or unrecognized
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df = df[df[TARGET_COL].isin(LABEL_MAPPING.keys())]

    y = df[TARGET_COL].map(LABEL_MAPPING).values
    X = df.drop(columns=[TARGET_COL])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = CreditDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    from pathlib import Path
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv"
    if sample_csv.exists():
        print(f"[TEST] Testing preprocessing on {sample_csv}...")
        X_tr, X_te, y_tr, y_te, prep = prepare_train_test_data(sample_csv)
        print(f"[TEST] X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
        print(f"[TEST] Features count: {len(prep.feature_names)}")
