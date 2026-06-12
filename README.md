# Predictive Analysis for Churn Management and Customer Retention

A comparative machine-learning study on predicting customer churn in the telecom industry, evaluating five classifiers with and without class-imbalance handling, and interpreting the results using explainable-AI techniques.

**Author:** Erika Mediratta
**Status:** Independent research paper (unpublished, IEEE format)

📄 **Read the paper:** [Predictive_Analysis_for_Churn_Management.pdf](https://github.com/Erikamediratta/Telecom-churn-prediction/blob/main/Erika_Mediratta_Churn_Paper.pdf)
🔗 **Deployed app (related project):** [Customer-Churn-Prediction](https://github.com/Erikamediratta/Customer-Churn-Prediction)



---

## Overview

Customer churn — when a customer stops using a company's service — is a major cost driver in telecom, where acquiring a new customer is more expensive than retaining an existing one. This project uses machine learning to predict which customers are likely to churn, so retention efforts can be targeted before they leave.

The study compares **five classifiers** on the IBM Telco Customer Churn dataset, measures the effect of **two class-balancing techniques (SMOTE and ADASYN)**, and uses **SHAP and LIME** to explain the predictions.

## Dataset

- **Source:** IBM Telco Customer Churn — [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customer records, 21 attributes (demographics, account details, subscribed services)
- **Target:** `Churn` (Yes / No) — a binary classification problem
- **Class balance:** imbalanced — ~74% non-churn, ~26% churn

> The dataset is the public IBM/Kaggle Telco dataset and is included here for reproducibility. All credit to the original source above.

## Methodology

**Preprocessing**
- Converted blank `TotalCharges` values to `NaN` and dropped them (7,043 → 7,032 rows)
- Dropped the non-predictive `customerID` column and removed duplicate records (→ 7,010 rows, 20 columns)
- Label encoding for binary columns; one-hot encoding (`drop_first=True`) for multi-valued columns to avoid multicollinearity
- `StandardScaler` applied before distance-based models (KNN)

**Models**
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naïve Bayes

**Class-imbalance handling:** SMOTE and ADASYN (applied to training data)

**Evaluation:** Accuracy, Precision, Recall, F1-score, AUC-ROC, confusion matrices, and ROC-curve comparison — measured before and after balancing.

**Explainability:** SHAP (global feature importance) and LIME (individual-prediction explanations).

**Split:** 80/20 train/test, `random_state=42` for reproducibility.

## Key Results

| Model | Accuracy (raw) | AUC (raw) | Accuracy (SMOTE) | AUC (SMOTE) |
|-------|:---:|:---:|:---:|:---:|
| Logistic Regression | 82% | 0.85 | 76% | 0.81 |
| Random Forest | 81% | 0.83 | **77%** | **0.82** |
| SVM | 82% | 0.80 | 69% | 0.82 |
| KNN | 77% | 0.74 | 70% | 0.76 |
| Naïve Bayes | 65% | 0.82 | 61% | 0.81 |

**Takeaways**
- On the **raw imbalanced** data, Logistic Regression scored highest on accuracy (82%) and AUC (0.85) — but this partly reflects the model favouring the majority (non-churn) class.
- After **balancing (SMOTE / ADASYN)**, accuracy dipped slightly, but **recall on the churn class improved across most models** — meaning fewer at-risk customers are missed, which is what matters most for retention.
- **Random Forest after balancing** gave the most robust, well-rounded performance (Accuracy 77%, F1 0.78, AUC 0.82), making it the practical choice.
- KNN underperformed on this high-dimensional, one-hot-encoded data (curse of dimensionality).
- **Most influential features** (SHAP / feature importance): `TotalCharges`, `tenure`, `MonthlyCharges`, and `InternetService_Fiber optic` — newer customers and fiber-optic subscribers were more likely to churn.

## Repository Structure

```
.
├── README.md
├── churn_analysis.ipynb        # full analysis: preprocessing, models, evaluation, SHAP/LIME
├── telco_customer_churn.csv    # IBM Telco dataset (Kaggle)
└── Predictive_Analysis_for_Churn_Management.pdf   # the research paper
```

## Run It Locally

```bash
# 1. Clone
git clone https://github.com/Erikamediratta/Telecom-churn-prediction.git
cd Telecom-churn-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn shap lime matplotlib seaborn jupyter

# 4. Launch the notebook
jupyter notebook churn_analysis.ipynb
```

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn (SMOTE / ADASYN)` · `SHAP` · `LIME` · `Matplotlib` · `Seaborn`

## Future Work

- Add Decision Tree and Gradient Boosting models
- Hyperparameter tuning and feature selection
- Explore further imbalance-handling techniques to improve robustness and generalisation

## References

1. S. K. Wagh et al., "Customer churn prediction in telecom sector using machine learning techniques," *Results in Control and Optimization*, vol. 14, 2024.
2. M. Yuvalı, B. Yaman, Ö. Tosun, "Classification comparison of machine learning algorithms using two independent CAD datasets," *Mathematics*, vol. 10, no. 3, 2022.
3. IBM, "Telco Customer Churn Dataset," Kaggle, 2018.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, 2011.

---

*Author: Erika Mediratta · B.Sc. (Hons) Computer Science, University of Delhi*
