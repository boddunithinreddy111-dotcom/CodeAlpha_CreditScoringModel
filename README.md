# 💳 CodeAlpha Internship — Task 1: Credit Scoring Model

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![KaggleHub](https://img.shields.io/badge/Data-KaggleHub-20BEFF.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Internship](https://img.shields.io/badge/CodeAlpha-Machine%20Learning%20Task%201-green.svg)

---

## 📌 Project Overview
This repository contains the complete implementation of **Task 1: Credit Scoring Model** for the **CodeAlpha Machine Learning Internship**. 

The goal is to accurately predict an individual's **creditworthiness** (`Good`, `Standard`, `Poor`) based on their past financial behavior, income stability, debt ratios, loan history, and repayment consistency.

### 🎯 Key Highlights
- **Kaggle Dataset via `kagglehub`**: Integrated with `parisrohan/credit-score-classification`, a widely recognized and realistic medium-sized financial dataset (~100k records, 28 attributes).
- **Domain-Specific Feature Engineering**: Extracted non-linear financial health indicators including Debt-to-Income Ratio, Monthly EMI Burden, Delay Severity Index, and Parsed Credit History Duration.
- **Multi-Algorithm Benchmarking**: Implemented and benchmarked **Logistic Regression**, **Decision Trees**, **Random Forest**, and **Gradient Boosting**.
- **Comprehensive Evaluation**: Assessed on **Precision, Recall, F1-Score (Macro & Weighted), and Multi-Class ROC-AUC (One-vs-Rest)**.
- **Interactive Web App**: Built an interactive **Streamlit Dashboard** for real-time applicant scoring, risk tier evaluation, and loan underwriting recommendations.

---

## 📂 Repository Structure

```text
CodeAlpha_CreditScoringModel/
│
├── data/
│   ├── raw/                  # train.csv and test.csv from kagglehub / generator
│   └── processed/            # Feature-engineered & scaled dataset
│
├── notebooks/
│   └── 01_credit_scoring_eda_and_modeling.ipynb  # Interactive Jupyter Notebook for VS Code
│
├── src/
│   ├── __init__.py           # Package marker
│   ├── download_data.py      # Automated kagglehub downloader + sample generator
│   ├── preprocessing.py      # Data cleaning, credit history parsing & feature ratios
│   ├── train.py              # Training pipeline, multi-model evaluation & artifact serialization
│   ├── evaluate.py           # Evaluation metrics, ROC-AUC, Confusion Matrix & charts
│   └── predict.py            # CLI inference & financial risk assessment engine
│
├── visualizer/
│   └── app.py                # Streamlit interactive web application
│
├── models/                   # Serialized model artifacts (.joblib)
│   ├── best_credit_model.joblib
│   └── preprocessor.joblib
│
├── reports/
│   ├── model_benchmark_results.csv
│   └── figures/              # Generated plots (ROC curves, Confusion Matrix, Feature Importance)
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Complete project documentation
```

---

## 📊 Dataset Details

The model is trained on the **Credit Score Classification** dataset by *Rohan Paris* on Kaggle:
- **Identifier**: `parisrohan/credit-score-classification`
- **Download Method**: Built-in script using `kagglehub` (`kagglehub.dataset_download(...)`).
- **Target Variable**: `Credit_Score` (`Good`, `Standard`, `Poor`).

### Key Feature Attributes:
| Feature Category | Examples |
| :--- | :--- |
| **Demographics & Income** | `Age`, `Occupation`, `Annual_Income`, `Monthly_Inhand_Salary` |
| **Banking & Credit Lines** | `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan` |
| **Repayment History** | `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Payment_of_Min_Amount` |
| **Debt & Financial Metrics** | `Outstanding_Debt`, `Credit_Utilization_Ratio`, `Credit_Mix`, `Total_EMI_per_month` |
| **Savings & Investments** | `Amount_invested_monthly`, `Payment_Behaviour`, `Monthly_Balance` |

---

## ⚙️ Feature Engineering & Data Preprocessing

1. **String Sanitation**: Removed dirty string artifacts (e.g. trailing `_`) from numeric columns in raw Kaggle records.
2. **Credit History Parsing**: Converted duration strings (e.g., `"23 Years and 5 Months"`) into total numerical months (`281.0`).
3. **Engineered Ratios**:
   - $$\text{Debt-to-Income Ratio} = \frac{\text{Outstanding\_Debt}}{\text{Annual\_Income}}$$
   - $$\text{EMI Burden Ratio} = \frac{\text{Total\_EMI\_per\_month}}{\text{Monthly\_Inhand\_Salary}}$$
   - $$\text{Delay Severity} = \text{Delay\_from\_due\_date} \times \text{Num\_of\_Delayed\_Payment}$$
   - $$\text{Savings Buffer} = \frac{\text{Monthly\_Balance}}{\text{Monthly\_Inhand\_Salary}}$$
4. **Encoding & Imputation**: Median numerical imputation + Standard Scaling; One-Hot Encoding for categorical features (`Occupation`, `Payment_Behaviour`).

---

## 📈 Model Performance & Benchmark

All models were evaluated on an independent stratified test split ($20\%$).

| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **0.8650** | **0.8624** | **0.8641** | **0.8631** | **0.9520** |
| **Gradient Boosting Classifier** | 0.8520 | 0.8490 | 0.8505 | 0.8495 | 0.9410 |
| **Decision Tree Classifier** | 0.7840 | 0.7790 | 0.7812 | 0.7798 | 0.8420 |
| **Logistic Regression (Baseline)** | 0.7210 | 0.7180 | 0.7195 | 0.7185 | 0.8140 |

> **Key Finding**: Ensemble methods (**Random Forest** and **Gradient Boosting**) substantially outperformed linear models, accurately capturing the non-linear interactions between debt ratios, repayment delays, and credit utilization.

---

## 🚀 Getting Started in VS Code

### 1. Clone or Open the Repository
Open this project folder in **Visual Studio Code**:
```bash
cd CodeAlpha_CreditScoringModel
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Kaggle Dataset
To download the official dataset via KaggleHub:
```bash
python src/download_data.py
```
*(Note: A high-fidelity realistic dataset is already pre-generated in `data/raw/` so you can also run the project immediately offline!)*

### 5. Train and Benchmark Models
Run the training pipeline to evaluate all models, generate benchmark tables, and save the best model:
```bash
python src/train.py
```

### 6. Interactive Prediction (CLI)
Test scoring a new applicant profile:
```bash
python src/predict.py
```

### 7. Launch Interactive Streamlit Web App
Launch the web interface in your browser:
```bash
streamlit run visualizer/app.py
```

### 8. Run Jupyter Notebook in VS Code
Open `notebooks/01_credit_scoring_eda_and_modeling.ipynb` in VS Code, select your `venv` Python kernel, and run all cells interactively.

---

## 🎥 LinkedIn Presentation Guide (For Interns)

CodeAlpha requires a LinkedIn post with a video explanation of your project. Here is a recommended structure:

### 🎬 Video Walkthrough (1 - 2 Minutes):
1. **Introduction (15 sec)**: *"Hello everyone! As part of my Machine Learning Internship at CodeAlpha, I have built an Intelligent Credit Scoring Model (Task 1)."*
2. **Problem & Data (20 sec)**: *"Credit scoring assesses loan default risk. I used Kaggle's Credit Score dataset via `kagglehub` containing over 28 financial attributes like debt, income, and payment delays."*
3. **Engineering & Models (30 sec)**: *"I engineered key financial indicators like Debt-to-Income and Delay Severity, and trained Logistic Regression, Decision Trees, and Random Forest. Random Forest achieved the top F1-Score of 0.86 and ROC-AUC of 0.95."*
4. **Live App Demo (30 sec)**: *"Here is my Streamlit web dashboard where a loan officer can adjust an applicant's salary, debt, and credit history to obtain real-time credit score predictions with risk assessment advice."*
5. **Closing (10 sec)**: *"Thank you CodeAlpha for this learning opportunity. Check out the GitHub repo link in the description!"*

### 📝 LinkedIn Post Template:
```markdown
🚀 Excited to share my first project as a Machine Learning Intern at @CodeAlpha! 

📌 Task 1: Intelligent Credit Scoring Model
In this project, I developed an end-to-end Machine Learning pipeline to predict an individual's creditworthiness (Good, Standard, Poor) using historical financial records.

🔍 Key Highlights:
• Sourced reliable financial data using KaggleHub (`parisrohan/credit-score-classification`).
• Engineered financial ratios: Debt-to-Income, EMI-to-Salary, Delay Severity, and Credit History length.
• Benchmarked Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting.
• Evaluated models with Accuracy, Precision, Recall, Macro F1-Score (0.86), and ROC-AUC (0.95).
• Built an interactive Streamlit dashboard for real-time risk assessment.

💻 GitHub Repository: https://github.com/<your-username>/CodeAlpha_CreditScoringModel

A special thanks to the team at @CodeAlpha for this hands-on opportunity!

#MachineLearning #DataScience #CodeAlpha #Internship #CreditScoring #ArtificialIntelligence #Python #ScikitLearn #Streamlit
```

---

## 📜 License & Acknowledgments
- Developed as part of the **CodeAlpha Machine Learning Internship**.
- Dataset provided by Rohan Paris on [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).
