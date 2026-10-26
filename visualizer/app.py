"""
Streamlit Web Dashboard for Credit Score Prediction - CodeAlpha Internship Task 1.
Run locally with: streamlit run visualizer/app.py
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_single_applicant, load_artifacts

st.set_page_config(
    page_title="Credit Scoring System | CodeAlpha ML Internship",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .score-card-good {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .score-card-standard {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .score-card-poor {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💳 Intelligent Credit Scoring & Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CodeAlpha Machine Learning Internship — Task 1: Supervised Creditworthiness Classification</div>', unsafe_allow_html=True)

# Sidebar - Applicant Parameters
st.sidebar.header("📋 Applicant Profile Inputs")

st.sidebar.subheader("👤 Personal & Professional")
age = st.sidebar.slider("Age (Years)", min_value=18, max_value=75, value=30)
occupation = st.sidebar.selectbox(
    "Occupation",
    ["Engineer", "Scientist", "Teacher", "Developer", "Doctor", "Lawyer", "Accountant", "Architect", "Entrepreneur", "Mechanic"]
)

st.sidebar.subheader("💰 Income & Banking")
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=5000.0, max_value=300000.0, value=65000.0, step=1000.0)
monthly_salary = st.sidebar.number_input("Monthly In-hand Salary ($)", min_value=400.0, max_value=25000.0, value=annual_income / 12 * 0.8, step=100.0)
num_bank_accounts = st.sidebar.slider("Number of Bank Accounts", min_value=1, max_value=12, value=3)
num_credit_cards = st.sidebar.slider("Number of Credit Cards", min_value=0, max_value=12, value=3)
interest_rate = st.sidebar.slider("Average Interest Rate (%)", min_value=1, max_value=35, value=12)

st.sidebar.subheader("📊 Debt & Credit Repayment History")
num_loans = st.sidebar.slider("Number of Active Loans", min_value=0, max_value=10, value=1)
delay_days = st.sidebar.slider("Average Days Delayed Past Due Date", min_value=0, max_value=60, value=3)
delayed_payments = st.sidebar.slider("Count of Delayed Payments in Last Year", min_value=0, max_value=30, value=1)
outstanding_debt = st.sidebar.number_input("Total Outstanding Debt ($)", min_value=0.0, max_value=50000.0, value=1200.0, step=100.0)
utilization = st.sidebar.slider("Credit Card Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)

credit_history_years = st.sidebar.slider("Credit History Length (Years)", min_value=0, max_value=35, value=10)
credit_mix = st.sidebar.selectbox("Credit Mix Quality", ["Good", "Standard", "Bad"], index=0)
payment_min = st.sidebar.selectbox("Pays Only Minimum Amount?", ["No", "Yes"], index=0)

total_emi = st.sidebar.number_input("Total Monthly EMI ($)", min_value=0.0, max_value=5000.0, value=250.0, step=50.0)
invested_monthly = st.sidebar.number_input("Monthly Investments ($)", min_value=0.0, max_value=5000.0, value=300.0, step=50.0)
monthly_balance = st.sidebar.number_input("Monthly Cash Balance Remaining ($)", min_value=0.0, max_value=20000.0, value=2200.0, step=50.0)
payment_behaviour = st.sidebar.selectbox(
    "Observed Payment Behaviour",
    [
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "High_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "High_spent_Large_value_payments"
    ]
)

# Main Application Layout
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("Applicant Financial Snapshot")
    debt_to_income = (outstanding_debt / annual_income) * 100
    emi_to_salary = (total_emi / monthly_salary) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Debt-to-Income", f"{debt_to_income:.1f}%")
    m2.metric("EMI Burden", f"{emi_to_salary:.1f}%")
    m3.metric("Utilization", f"{utilization:.1f}%")
    m4.metric("Credit Age", f"{credit_history_years} Yrs")

    applicant_data = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_days,
        "Num_of_Delayed_Payment": delayed_payments,
        "Changed_Credit_Limit": 5.0,
        "Num_Credit_Inquiries": 2,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": utilization,
        "Credit_History_Age": f"{credit_history_years} Years and 0 Months",
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested_monthly,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }

    st.write("---")
    predict_btn = st.button("🚀 Evaluate Applicant Creditworthiness", type="primary", use_container_width=True)

with col_right:
    st.subheader("Model Evaluation Results")

    if predict_btn:
        try:
            result = predict_single_applicant(applicant_data)
            score = result["Predicted_Score"]
            probs = result["Probabilities"]

            if score == "Good":
                card_class = "score-card-good"
                badge = "🟢 GOOD CREDIT"
            elif score == "Standard":
                card_class = "score-card-standard"
                badge = "🟡 STANDARD CREDIT"
            else:
                card_class = "score-card-poor"
                badge = "🔴 POOR CREDIT"

            st.markdown(f"""
            <div class="{card_class}">
                <h3>{badge}</h3>
                <p><strong>Predicted Classification:</strong> {score}</p>
                <p>{result['Recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.subheader("Class Probability Breakdown")
            prob_df = pd.DataFrame({
                "Credit Category": list(probs.keys()),
                "Model Confidence": [f"{v * 100:.1f}%" for v in probs.values()],
                "Raw Score": list(probs.values())
            })
            st.dataframe(prob_df[["Credit Category", "Model Confidence"]], use_container_width=True, hide_index=True)

            st.bar_chart(prob_df.set_index("Credit Category")["Raw Score"])

        except FileNotFoundError:
            st.warning("⚠️ Model weights not found! Please run `python src/train.py` first to train and generate the model artifacts.")
        except Exception as e:
            st.error(f"Error predicting: {e}")
    else:
        st.info("👈 Adjust applicant parameters in the left sidebar and click **'Evaluate Applicant Creditworthiness'** to get real-time AI credit assessment.")

st.write("---")
st.markdown("""
**CodeAlpha Internship Submission Notes:**
- **Objective:** Predict individual creditworthiness from past financial behavior.
- **Algorithms Implemented:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Key Metrics:** Precision, Recall, F1-Score, ROC-AUC.
""")
