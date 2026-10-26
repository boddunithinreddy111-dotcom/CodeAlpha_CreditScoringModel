"""
Data Download and Generation Module for Credit Scoring Model.
Downloads the Credit Score Classification dataset using kagglehub,
with a robust local synthetic fallback generator for offline testing.
"""

import os
import shutil
import random
import csv
from pathlib import Path

DATASET_NAME = "parisrohan/credit-score-classification"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_from_kagglehub():
    """
    Downloads the dataset from Kaggle via kagglehub library
    and copies train.csv and test.csv into data/raw/.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_dest = RAW_DATA_DIR / "train.csv"
    test_dest = RAW_DATA_DIR / "test.csv"

    if train_dest.exists() and train_dest.stat().st_size > 1024 * 1024:
        print(f"[INFO] Dataset already present at: {train_dest}")
        return train_dest, test_dest

    print(f"[INFO] Attempting to download '{DATASET_NAME}' via kagglehub...")
    try:
        import kagglehub

        download_path = kagglehub.dataset_download(DATASET_NAME)
        print(f"[SUCCESS] Downloaded to cache path: {download_path}")

        src_train = Path(download_path) / "train.csv"
        src_test = Path(download_path) / "test.csv"

        if src_train.exists():
            shutil.copy(src_train, train_dest)
            print(f"[COPIED] train.csv -> {train_dest}")
        if src_test.exists():
            shutil.copy(src_test, test_dest)
            print(f"[COPIED] test.csv -> {test_dest}")

        return train_dest, test_dest
    except Exception as e:
        print(f"[WARNING] kagglehub download failed or unavailable ({e}).")
        print("[INFO] Falling back to generating a realistic synthetic dataset for local development...")
        generate_sample_dataset(train_dest, n_samples=3000)
        generate_sample_dataset(test_dest, n_samples=600, is_test=True)
        return train_dest, test_dest


def generate_sample_dataset(output_path: Path, n_samples: int = 3000, is_test: bool = False):
    """
    Generates a realistic synthetic credit scoring dataset conforming exactly
    to the 28-column schema of parisrohan/credit-score-classification.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    occupations = [
        "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
        "Lawyer", "Media_Manager", "Doctor", "Journalist", "Accountant",
        "Architect", "Musician", "Mechanic", "Writer"
    ]
    loan_types = [
        "Auto Loan, Credit-Builder Loan",
        "Home Equity Loan, Mortgage Loan",
        "Personal Loan",
        "Student Loan",
        "Debt Consolidation Loan",
        "Not Specified"
    ]
    credit_mixes = ["Good", "Standard", "Bad", "_"]
    payment_behaviours = [
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "High_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "High_spent_Large_value_payments"
    ]

    fieldnames = [
        "ID", "Customer_ID", "Month", "Name", "Age", "SSN",
        "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Type_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age", "Payment_of_Min_Amount", "Total_EMI_per_month",
        "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance"
    ]
    if not is_test:
        fieldnames.append("Credit_Score")

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, n_samples + 1):
            cust_id = f"CUS_0x{random.randint(1000, 9999):x}"
            record_id = f"0x{i:04x}"
            age = random.randint(18, 65)
            # Occasional string artifacts common in the original raw dataset
            age_str = str(age) if random.random() > 0.05 else f"{age}_"

            income = round(random.uniform(15000, 180000), 2)
            income_str = str(income) if random.random() > 0.05 else f"{income}_"

            monthly_salary = round(income / 12 * random.uniform(0.70, 0.90), 2)
            num_bank_accounts = random.randint(1, 10)
            num_credit_cards = random.randint(1, 11)
            interest_rate = random.randint(1, 35)
            num_loans = random.randint(0, 9)
            delay_days = max(0, int(random.gauss(15, 12)))
            delayed_payments = min(delay_days, random.randint(0, 25))
            delayed_payments_str = str(delayed_payments) if random.random() > 0.05 else f"{delayed_payments}_"
            limit_change = round(random.uniform(-5.0, 28.0), 2)
            inquiries = random.randint(0, 15)

            outstanding_debt = round(random.uniform(100, 4500), 2)
            debt_str = str(outstanding_debt) if random.random() > 0.05 else f"{outstanding_debt}_"
            utilization = round(random.uniform(20.0, 45.0), 2)

            hist_years = random.randint(1, 30)
            hist_months = random.randint(0, 11)
            history_str = f"{hist_years} Years and {hist_months} Months"

            min_amt_paid = "Yes" if random.random() > 0.4 else "No"
            emi = round(random.uniform(20, 800), 2)
            invested = round(random.uniform(20, 1200), 2)
            balance = max(0.0, round(monthly_salary - emi - invested + random.uniform(-100, 200), 2))

            credit_mix = random.choice(credit_mixes)

            # Rule-based credit score assignment to preserve real statistical signals
            score_metric = (
                (income / 10000.0) * 1.5
                - (outstanding_debt / 500.0) * 2.0
                - (delay_days / 5.0) * 1.8
                - (delayed_payments / 3.0) * 1.5
                - (num_credit_cards / 2.0)
                + (hist_years * 0.8)
                - (interest_rate / 4.0)
            )

            if score_metric > 8:
                credit_score = "Good"
            elif score_metric > -2:
                credit_score = "Standard"
            else:
                credit_score = "Poor"

            row = {
                "ID": record_id,
                "Customer_ID": cust_id,
                "Month": random.choice(["January", "February", "March", "April", "May", "June"]),
                "Name": f"Customer_{i}",
                "Age": age_str,
                "SSN": f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
                "Occupation": random.choice(occupations),
                "Annual_Income": income_str,
                "Monthly_Inhand_Salary": monthly_salary,
                "Num_Bank_Accounts": num_bank_accounts,
                "Num_Credit_Card": num_credit_cards,
                "Interest_Rate": interest_rate,
                "Num_of_Loan": str(num_loans),
                "Type_of_Loan": random.choice(loan_types),
                "Delay_from_due_date": delay_days,
                "Num_of_Delayed_Payment": delayed_payments_str,
                "Changed_Credit_Limit": limit_change,
                "Num_Credit_Inquiries": inquiries,
                "Credit_Mix": credit_mix,
                "Outstanding_Debt": debt_str,
                "Credit_Utilization_Ratio": utilization,
                "Credit_History_Age": history_str,
                "Payment_of_Min_Amount": min_amt_paid,
                "Total_EMI_per_month": emi,
                "Amount_invested_monthly": str(invested),
                "Payment_Behaviour": random.choice(payment_behaviours),
                "Monthly_Balance": str(balance),
            }
            if not is_test:
                row["Credit_Score"] = credit_score

            writer.writerow(row)

    print(f"[CREATED] Synthetic dataset saved to {output_path} ({n_samples} records).")


if __name__ == "__main__":
    download_from_kagglehub()
