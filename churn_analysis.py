"""
Churn Prediction — Logistic Regression with JSON export for web dashboard.
Run: python churn_analysis.py
Requires: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load & Preprocess ─────────────────────────────────────────────────────

df = pd.read_csv('churn_data.csv')
df = pd.get_dummies(df, columns=['SubscriptionType'], drop_first=False)

FEATURES = [
    'Age', 'MonthlyCharges', 'TotalUsageHours', 'SupportTickets',
    'ContractDuration_Months', 'TenureMonths', 'NumProducts',
    'SubscriptionType_Premium', 'SubscriptionType_Standard',
]
X = df[FEATURES]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 2. Train ─────────────────────────────────────────────────────────────────

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_s, y_train)

# ── 3. Evaluate ──────────────────────────────────────────────────────────────

y_pred   = model.predict(X_test_s)
y_proba  = model.predict_proba(X_test_s)[:, 1]
report   = classification_report(y_test, y_pred, output_dict=True)
cm       = confusion_matrix(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)

print(f"\nAccuracy : {report['accuracy']:.3f}")
print(f"AUC-ROC  : {auc:.3f}")
print(f"Precision (churn): {report['1']['precision']:.3f}")
print(f"Recall    (churn): {report['1']['recall']:.3f}")
print(f"F1        (churn): {report['1']['f1-score']:.3f}")

# ── 4. Feature Importance ────────────────────────────────────────────────────

feature_importance = pd.Series(model.coef_[0], index=FEATURES).sort_values()

# ── 5. Visualizations ────────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix — Churn Prediction')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# ── 6. Export JSON for web dashboard ─────────────────────────────────────────

os.makedirs('data', exist_ok=True)

# Model performance
perf = {
    'accuracy':  round(report['accuracy'], 4),
    'auc_roc':   round(auc, 4),
    'precision': round(report['1']['precision'], 4),
    'recall':    round(report['1']['recall'], 4),
    'f1':        round(report['1']['f1-score'], 4),
    'train_size': int(len(X_train)),
    'test_size':  int(len(X_test)),
    'churn_rate': round(y.mean(), 4),
    'n_customers': int(len(df)),
}
with open('data/model_performance.json', 'w') as f:
    json.dump(perf, f, indent=2)

# Confusion matrix
cm_data = {
    'tn': int(cm[0][0]), 'fp': int(cm[0][1]),
    'fn': int(cm[1][0]), 'tp': int(cm[1][1]),
}
with open('data/confusion_matrix.json', 'w') as f:
    json.dump(cm_data, f, indent=2)

# Feature importance
fi = [{'feature': k, 'coefficient': round(v, 4)}
      for k, v in feature_importance.items()]
with open('data/feature_importance.json', 'w') as f:
    json.dump(fi, f, indent=2)

# Churn by segment
for col, fname in [
    ('SubscriptionType', 'churn_by_subscription'),
    ('ContractDuration_Months', 'churn_by_contract'),
]:
    raw_col = col if col in df.columns else col
    if col == 'SubscriptionType':
        tmp = pd.read_csv('churn_data.csv')
    else:
        tmp = df.copy()
        tmp['ContractDuration_Months'] = df['ContractDuration_Months']

    if col == 'SubscriptionType':
        grp = tmp.groupby('SubscriptionType')['Churn'].agg(
            total='count', churned='sum'
        ).reset_index()
        grp['churn_rate'] = (grp['churned'] / grp['total']).round(4)
        grp = grp.rename(columns={'SubscriptionType': 'segment'})
    else:
        grp = tmp.groupby('ContractDuration_Months')['Churn'].agg(
            total='count', churned='sum'
        ).reset_index()
        grp['churn_rate'] = (grp['churned'] / grp['total']).round(4)
        grp['label'] = grp['ContractDuration_Months'].map(
            {1: 'Monthly', 12: 'Annual', 24: '2-Year'}
        )
        grp = grp.rename(columns={'ContractDuration_Months': 'segment'})

    with open(f'data/{fname}.json', 'w') as f:
        json.dump(grp.to_dict(orient='records'), f, indent=2)

# Support tickets vs churn rate
tmp2 = pd.read_csv('churn_data.csv')
tmp2['ticket_bucket'] = pd.cut(tmp2['SupportTickets'],
    bins=[-1, 0, 2, 4, 6, 20],
    labels=['0', '1-2', '3-4', '5-6', '7+']
)
tickets_grp = tmp2.groupby('ticket_bucket', observed=True)['Churn'].agg(
    total='count', churned='sum'
).reset_index()
tickets_grp['churn_rate'] = (tickets_grp['churned'] / tickets_grp['total']).round(4)
tickets_grp['ticket_bucket'] = tickets_grp['ticket_bucket'].astype(str)
with open('data/churn_by_tickets.json', 'w') as f:
    json.dump(tickets_grp.to_dict(orient='records'), f, indent=2)

# Monthly charges buckets
tmp2['charges_bucket'] = pd.cut(tmp2['MonthlyCharges'],
    bins=[0, 40, 60, 80, 100, 120, 200],
    labels=['$0-40', '$40-60', '$60-80', '$80-100', '$100-120', '$120+']
)
charges_grp = tmp2.groupby('charges_bucket', observed=True)['Churn'].agg(
    total='count', churned='sum'
).reset_index()
charges_grp['churn_rate'] = (charges_grp['churned'] / charges_grp['total']).round(4)
charges_grp['charges_bucket'] = charges_grp['charges_bucket'].astype(str)
with open('data/churn_by_charges.json', 'w') as f:
    json.dump(charges_grp.to_dict(orient='records'), f, indent=2)

print("\nJSON exports saved to data/")
print("Analysis complete.")
