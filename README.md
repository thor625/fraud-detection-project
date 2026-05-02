# Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions, deployed on AWS with a React dashboard.

## Project Overview

Built using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions made by European cardholders in September 2013.

## Tech Stack

- **ML:** Python, Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Cloud:** AWS S3, SageMaker, EC2, API Gateway, DynamoDB, CloudFront
- **API:** FastAPI
- **Frontend:** React
- **Other:** Jupyter, boto3, pandas, matplotlib, seaborn

## Project Structure

```
fraud-detection-project/
├── notebooks/
│   ├── 01_eda.ipynb              ← exploratory data analysis
│   ├── 02_preprocessing.ipynb    ← scaling, SMOTE, train/test split
│   └── 03_training.ipynb         ← model training and evaluation
├── src/
│   ├── lib/
│   │   └── connectors.py         ← shared AWS utilities (S3, DynamoDB)
│   └── api/
│       ├── main.py               ← FastAPI routes
│       └── model.py              ← prediction logic
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Exploratory Data Analysis

### Dataset

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Legitimate transactions | 284,315 |
| Fraud rate | 0.173% |
| Features | 28 PCA components (V1-V28) + Time + Amount |

### Key Findings

**1. Severe class imbalance**

99.827% of transactions are legitimate and only 0.173% are fraudulent. A naive model 
that predicts "legitimate" for every transaction achieves 99.8% accuracy but catches 
zero fraud. This makes accuracy a meaningless metric — we evaluate using precision, 
recall, and AUC-PR instead. SMOTE is applied to the training set to address this imbalance.

**2. Fraud concentrates at lower transaction amounts**

Legitimate transactions range up to $25,000 with most clustering under $2,500. 
Fraudulent transactions are far more constrained — nearly all fall under $1,500, 
with the majority under $500. This is consistent with real-world fraud patterns 
where fraudsters make small test transactions to verify a stolen card before 
attempting larger purchases.

**3. V17 is the strongest negative predictor of fraud**

Legitimate transactions cluster tightly around 0 on the V17 axis (-5 to +5). 
Fraudulent transactions spread widely into very negative territory, ranging from 
-25 to +8. A low V17 value is a strong signal of fraud.

**4. V11 is the strongest positive predictor of fraud**

Legitimate transactions follow a normal distribution on V11 centered around 0 
(-4 to +10). Fraudulent transactions shift significantly to the right, clustering 
between +2 and +8. A high V11 value is a strong signal of fraud.

**5. Multiple strong negative correlators beyond V17**

The correlation chart reveals V14, V12, V10, V16, V3 and V7 also carry strong 
negative correlations with fraud (all above -0.15). On the positive side V4 and 
V11 are the strongest positive predictors. These features will likely dominate 
XGBoost feature importance after training.

## Preprocessing

The raw dataset required three preprocessing steps before training:

### 1. Feature scaling
V1-V28 are already scaled by PCA. However Amount ($0-$25,000) and Time 
(0-172,000 seconds) are on completely different scales to the PCA components. 
StandardScaler was applied to both, bringing all features onto a consistent scale 
so no single feature dominates due to magnitude alone.

### 2. Train/test split
The dataset was split 80/20 using stratified sampling to preserve the original 
0.173% fraud ratio in both sets:

| Set | Rows | Fraud rate |
|---|---|---|
| Training | 227,845 | 0.173% |
| Test | 56,962 | 0.172% |

### 3. SMOTE (training set only)
The severe class imbalance (394 fraud vs 227,451 legitimate in the training set) 
was addressed using SMOTE (Synthetic Minority Oversampling Technique). SMOTE 
generates synthetic fraud samples by interpolating between nearest neighbour fraud 
transactions in the PCA-defined feature space — rather than simply duplicating 
existing examples which would cause overfitting.

SMOTE was applied to the training set only. The test set remains imbalanced at 
0.172% fraud to accurately simulate real-world conditions.

| | Legitimate | Fraud |
|---|---|---|
| Before SMOTE | 227,451 | 394 |
| After SMOTE | 227,451 | 227,451 |

---

## Model Training

Three models were trained and evaluated on the held-out test set. Since accuracy 
is meaningless for imbalanced data (a model predicting legitimate every time scores 
99.8%), models are evaluated on AUC-PR and fraud class precision/recall.

### Results

| Model | AUC-ROC | AUC-PR | Fraud precision | Fraud recall | False alarms |
|---|---|---|---|---|---|
| Logistic Regression | 0.9698 | 0.7249 | 0.06 | 0.92 | 1,458 |
| Random Forest | 0.9688 | 0.8678 | 0.82 | 0.82 | 17 |
| **XGBoost** | **0.9792** | **0.8774** | **0.73** | **0.89** | **32** |

### Why XGBoost wins

XGBoost achieves the highest AUC-PR (0.8774) and AUC-ROC (0.9792), catching 87 
out of 98 fraud cases in the test set with only 32 false alarms. While Random 
Forest produces fewer false alarms (17), XGBoost catches 7 more fraud cases — 
a worthwhile tradeoff in a real fraud detection system where missing fraud is 
more costly than investigating a false alarm.

Logistic Regression catches the most fraud (90 cases) but raises 1,458 false 
alarms — completely impractical for production use.

### Feature importance

V14 dominates XGBoost feature importance with a score of 0.59 — nearly 10x 
more important than the next feature (V4 at 0.06). This aligns with EDA findings 
where V14 showed one of the strongest negative correlations with fraud (-0.30). 
V12 and V17 also appear in the top features, confirming the EDA analysis.

The full narrative:
- EDA identified V17 and V14 as the strongest negative correlators with fraud
- XGBoost training confirmed V14 as the dominant predictive feature
- This consistency between exploratory analysis and model behaviour indicates 
  the model is learning genuine fraud patterns rather than overfitting

### Why not accuracy?

A naive model predicting legitimate for every transaction scores 99.8% accuracy 
but catches zero fraud. AUC-PR measures performance across all classification 
thresholds focusing solely on the minority class — making it the correct metric 
for severely imbalanced datasets like this one.
