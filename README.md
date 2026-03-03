{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f1666728-008a-4fbf-898a-9cfa11a0abf2",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid character '├' (U+251C) (650541159.py, line 14)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[1], line 14\u001b[1;36m\u001b[0m\n\u001b[1;33m    ├── archive/\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid character '├' (U+251C)\n"
     ]
    }
   ],
   "source": [
    "# Loan Approval Prediction — ML Project\n",
    "\n",
    "## Overview\n",
    "An end-to-end Machine Learning project to predict loan approval status\n",
    "based on applicant details like income, credit history, loan amount, and more.\n",
    "\n",
    "## Dataset\n",
    "- 614 training samples, 12 features\n",
    "- Source: Analytics Vidhya Loan Prediction Dataset\n",
    "\n",
    "## Project Structure\n",
    "```\n",
    "loan_approval_project/\n",
    "├── archive/\n",
    "│   ├── train_u6lujuX_CVtuZ9i.csv\n",
    "│   └── test_Y3wMUE5_7gLdaTN.csv\n",
    "├── Loan_approval_EDA.ipynb\n",
    "├── Loan_Approval_Phase2.ipynb\n",
    "├── Loan_Approval_Phase3.ipynb\n",
    "├── Loan_Approval_Phase4.ipynb\n",
    "├── requirements.txt\n",
    "└── README.md\n",
    "```\n",
    "\n",
    "## Key Steps\n",
    "- Exploratory Data Analysis with 12+ visualizations\n",
    "- Feature Engineering (TotalIncome, EMI, LoanIncomeRatio and more)\n",
    "- Statistical Validation (Chi-square, T-test, IQR outlier detection)\n",
    "- ML Pipeline using sklearn ColumnTransformer\n",
    "- Models: Logistic Regression, Random Forest, XGBoost\n",
    "- Hyperparameter Tuning using GridSearchCV\n",
    "- Feature Importance analysis\n",
    "\n",
    "## Results\n",
    "| Model | Accuracy | F1 Score | ROC-AUC |\n",
    "|---|---|---|---|\n",
    "| Logistic Regression | 86.18% | 90.81% | 0.869 |\n",
    "| Random Forest | 85.37% | 89.66% | 0.874 |\n",
    "| XGBoost | 82.11% | 87.36% | 0.826 |\n",
    "\n",
    "## Key Finding\n",
    "Credit History is the single most important factor —\n",
    "applicants with credit history have 10x higher approval rate.\n",
    "\n",
    "## Tech Stack\n",
    "Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib\n",
    "\n",
    "## How To Run\n",
    "```bash\n",
    "# 1. Clone the repo\n",
    "git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git\n",
    "\n",
    "# 2. Create environment\n",
    "conda create -n loan_approval python=3.10\n",
    "conda activate loan_approval\n",
    "\n",
    "# 3. Install dependencies\n",
    "pip install -r requirements.txt\n",
    "\n",
    "# 4. Run notebooks in order\n",
    "# Phase 1 → EDA\n",
    "# Phase 2 → Statistical Validation\n",
    "# Phase 3 → Model Building\n",
    "# Phase 4 → Tuning + Feature Importance\n",
    "```\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "### Step 2 — Create .gitignore file\n",
    "Create `.gitignore` in your project folder:\n",
    "```\n",
    "*.pkl\n",
    "*.png\n",
    "__pycache__/\n",
    ".ipynb_checkpoints/\n",
    ".env"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed9376ca-09cc-4acc-89a7-37548fbb01ab",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Loan Approval Project",
   "language": "python",
   "name": "loan_approval"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
