# Loan Approval Prediction — ML Project

## Overview
An end-to-end Machine Learning project to predict loan approval status based on applicant details like income, credit history, loan amount, and property area.

## Dataset
- 614 training samples, 12 features
- Source: Analytics Vidhya Loan Prediction Dataset

## Project Structure
- Loan_approval_EDA.ipynb — Exploratory Data Analysis
- Loan_Approval_Phase2.ipynb — Statistical Validation + Preprocessing Pipeline
- Loan_Approval_Phase3.ipynb — Model Building
- Loan_Approval_Phase4.ipynb — Hyperparameter Tuning + Feature Importance
- requirements.txt — All dependencies

## Key Steps
- Exploratory Data Analysis with 12+ visualizations
- Feature Engineering — TotalIncome, EMI, LoanIncomeRatio, IncomePerPerson and more
- Statistical Validation using Chi-square test and T-test
- Outlier Detection using IQR method
- ML Pipeline using sklearn ColumnTransformer
- Models — Logistic Regression, Random Forest, XGBoost
- Hyperparameter Tuning using GridSearchCV with 5-fold Cross Validation
- Feature Importance analysis

## Results
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 86.18% | 90.81% | 0.869 |
| Random Forest | 85.37% | 89.66% | 0.874 |
| XGBoost After Tuning | 86.18% | 90.81% | 0.869 |

## Key Finding
Credit History is the single most important factor. Applicants with credit history have 10x higher approval rate than those without.

## Final Model
XGBoost after hyperparameter tuning — selected for lowest variance across 5-fold cross validation (std = 0.038) making it the most consistent and reliable model.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib

## How To Run
1. Clone the repo
2. Create environment — conda create -n loan_approval python=3.10
3. Activate — conda activate loan_approval
4. Install dependencies — pip install -r requirements.txt
5. Run notebooks in order Phase 1 to Phase 4
