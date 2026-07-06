# 🏦 Customer Churn Prediction System

### 🔗 [Live Demo](https://bank-customer-churn-prediction-p9iabpwp37pksd6fdpnjie.streamlit.app/)

An end-to-end machine learning application that predicts customer churn for an Indian banking dataset, and translates model output into tiered, actionable retention strategies — built to be read and used by a non-technical stakeholder, not just a data scientist.

---

## Overview

| | |
|---|---|
| **Problem** | Predict which bank customers are likely to churn, and prioritize retention effort by risk tier |
| **Data** | 100,000 synthetic customer records modeled on Indian banking demographics and behavior |
| **Model** | XGBoost, tuned via `RandomizedSearchCV`, with a custom decision threshold (not the default 0.5) |
| **Output** | Churn probability, risk tier, and specific retention actions — surfaced through a deployed Streamlit app |

---

## Data

- 100,000 rows, 13 original columns: `CustomerID`, `Age`, `Gender`, `State`, `Account_Type`, `Tenure_Years`, `Balance_INR`, `Num_Products`, `Has_Credit_Card`, `Is_Active_Member`, `Estimated_Salary_INR`, `Credit_Score`, `Churn`
- Target (`Churn`): imbalanced — most customers do not churn, which shaped the modeling choices below (class weighting, threshold tuning, F1-based tuning instead of accuracy)
- Synthetic dataset, generated to resemble Indian banking customer behavior — disclosed explicitly here since it is not real bank data

---

## Feature Engineering

Four domain-informed features were added on top of the raw columns, rather than feeding raw fields directly into the model:

| Feature | Logic | Why |
|---|---|---|
| `Balance_to_Salary` | `Balance_INR / Estimated_Salary_INR` | Raw balance means little without income context; this ratio captures relative financial exposure |
| `Tenure_NumProducts` | `Tenure_Years * Num_Products` | Interaction term — a long-tenure customer with many products behaves differently than either variable alone suggests |
| `Low_Credit_Score` | `Credit_Score < 600` (binary) | Converts a continuous score into a business-meaningful risk flag |
| `HighBalance_LowActivity` | `Balance_INR > 100,000` AND `Is_Active_Member == 0` (binary) | Flags a specific at-risk pattern: money parked in the account, but no engagement — this turned out to be the single strongest predictor in the final model |

---

## Pipeline (`02_Model_Training_Pipeline.ipynb`)

1. Engineer the four features above
2. Split features into numerical and categorical groups
3. 80/20 train-test split
4. Standard-scale numerical features (fit on train only)
5. One-hot encode categorical features — `Gender`, `State`, `Account_Type` (fit on train only, to avoid leakage)
6. Train and compare two baseline models: Random Forest and XGBoost, both untuned
7. Tune XGBoost via `RandomizedSearchCV` (30 iterations, 3-fold CV, scored on F1 — not accuracy, since the target is imbalanced), with `scale_pos_weight` set to the actual class ratio to correct for imbalance during training
8. Re-evaluate the tuned model at multiple decision thresholds instead of accepting the default 0.5, and select **0.45** as the operating threshold
9. Save the final model, scaler, encoder, and feature list for use in the Streamlit app

---

## Results

**Baseline comparison (default threshold, before tuning):**

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Random Forest | 64.7% | 0.55 | 0.36 | 0.44 |
| XGBoost (untuned) | 64.9% | 0.55 | 0.38 | 0.45 |

**Tuned XGBoost, threshold shifted from 0.50 → 0.45:**

| Metric | Value |
|---|---|
| Precision (Churn) | 0.50 |
| Recall (Churn) | 0.67 |
| F1 (Churn) | 0.57 |
| Overall Accuracy | 61.9% |

**Why accuracy goes down but the model is more useful:** the untuned models score higher raw accuracy but only catch 36–38% of actual churners — most churn cases slip through. Lowering the threshold to 0.45 trades some overall accuracy for catching two-thirds of churners (67% recall), which is the correct tradeoff for this business problem: the cost of missing a churning customer (lost revenue) is higher than the cost of a false alarm (a low-cost retention email to someone who wouldn't have churned anyway).

**Top features driving predictions (XGBoost feature importance):**
1. `Is_Active_Member` — 49.3%
2. `HighBalance_LowActivity` (engineered) — 27.8%
3. `Num_Products` — 6.0%
4. `Tenure_Years` — 3.2%
5. `Credit_Score` — 3.1%

Notably, the two features responsible for ~77% of the model's predictive power are activity-related, not demographic — reinforcing that *engagement*, not who the customer is, drives churn risk in this dataset.

---

## The Streamlit App (`app.py`)

- Sidebar-navigated, single-page dashboard with custom CSS
- **Live Prediction:** input form for an individual customer's attributes → churn probability, risk badge, and the specific contributing risk factors for that customer
- **Business Strategy view:** translates model output into tiered retention actions, framed around minimizing wasted marketing spend rather than blanket-targeting every flagged customer
- Deployed on Streamlit Community Cloud — link above

---

## Key Design Decisions & Honest Limitations

- **Threshold tuned to 0.45, not left at the default 0.5** — chosen via the precision-recall tradeoff described above, not arbitrarily
- **F1, not accuracy, used to guide tuning** — accuracy is misleading on imbalanced targets like this one, since a model that always predicts "no churn" would still score ~65% accuracy while catching zero real churners
- **Dataset is synthetic** — results describe how the model behaves on this generated data, not validated real-world bank churn rates
- **Precision at 0.50 means half of flagged "at-risk" customers will not actually churn** — this is a deliberate tradeoff for higher recall, and is stated explicitly in the app's business strategy view rather than left implicit

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `xgb_model.pkl`, `scaler.pkl`, `encoder.pkl`, and `model_features.pkl` in the same folder as `app.py` (included in this repo).

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `matplotlib` / `seaborn` (notebooks) · `plotly` (app) · `streamlit` · `joblib`

---

## Author

**Pooja Dhamale**
