"""
Churn Prediction — Logistic Regression with JSON export for web dashboard.
Run: python churn_analysis.py
Requires: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
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
plt.ylabel('Actual')
plt.xlabel('Predicted')
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

# ── Churn by segment ─────────────────────────────────────────────────────────
#
# The four segment files share one shape: {segment, total, churned, churn_rate},
# with `segment` always a label the dashboard can print. Emitting a different
# key per file — ticket_bucket, charges_bucket, a numeric segment plus a
# separate label — is what left three of the four charts without axis labels.

raw = pd.read_csv('churn_data.csv')


def por_segmento(serie, orden, fname):
    """Aggregate churn by an already-labelled series and write it out."""
    grp = (
        raw.groupby(serie, observed=True)['Churn']
        .agg(total='count', churned='sum')
        .reset_index()
    )
    grp.columns = ['segment', 'total', 'churned']
    grp['segment'] = grp['segment'].astype(str)
    grp['churn_rate'] = (grp['churned'] / grp['total']).round(4)
    grp['segment'] = pd.Categorical(grp['segment'], categories=orden, ordered=True)
    grp = grp.sort_values('segment')
    grp['segment'] = grp['segment'].astype(str)
    with open(f'data/{fname}.json', 'w') as f:
        json.dump(grp.to_dict(orient='records'), f, indent=2)


por_segmento(
    raw['SubscriptionType'],
    ['Basic', 'Standard', 'Premium'],
    'churn_by_subscription',
)

por_segmento(
    raw['ContractDuration_Months'].map({1: 'Monthly', 12: 'Annual', 24: '2-Year'}),
    ['Monthly', 'Annual', '2-Year'],
    'churn_by_contract',
)

por_segmento(
    pd.cut(raw['SupportTickets'], bins=[-1, 0, 2, 4, 6, 20],
           labels=['0', '1-2', '3-4', '5-6', '7+']),
    ['0', '1-2', '3-4', '5-6', '7+'],
    'churn_by_tickets',
)

por_segmento(
    pd.cut(raw['MonthlyCharges'], bins=[0, 40, 60, 80, 100, 120, 200],
           labels=['$0-40', '$40-60', '$60-80', '$80-100', '$100-120', '$120+']),
    ['$0-40', '$40-60', '$60-80', '$80-100', '$100-120', '$120+'],
    'churn_by_charges',
)

print("\nJSON exports saved to data/")

# ── 7. Results write-up ──────────────────────────────────────────────────────
#
# analysis_results.md used to be written by hand, so it drifted away from the
# code: it reported an accuracy from a dataset that no longer existed and no
# AUC at all. It is generated here for the same reason the JSONs are — a number
# nobody can regenerate is a number nobody should publish.

resumen = [
    "# Churn Prediction Analysis — Key Findings",
    "",
    "> Generated by `churn_analysis.py`. Do not edit by hand: re-run the script.",
    "",
    "## Model Performance (Logistic Regression)",
    "",
    "| Metric | Value |",
    "| :--- | ---: |",
    f"| AUC-ROC | {auc:.4f} |",
    f"| Accuracy | {report['accuracy']:.4f} |",
    f"| Precision (churn) | {report['1']['precision']:.4f} |",
    f"| Recall (churn) | {report['1']['recall']:.4f} |",
    f"| F1 (churn) | {report['1']['f1-score']:.4f} |",
    "",
    f"Dataset: {len(df):,} customers, {y.mean():.1%} churn rate. "
    f"Split {len(X_train):,} train / {len(X_test):,} test, stratified, `random_state=42`.",
    "",
    "## Confusion Matrix",
    "",
    "| | Predicted No Churn | Predicted Churn |",
    "| :--- | ---: | ---: |",
    f"| **Actual No Churn** | {cm[0][0]} | {cm[0][1]} |",
    f"| **Actual Churn** | {cm[1][0]} | {cm[1][1]} |",
    "",
    "## Feature Importance (Model Coefficients)",
    "",
    "Coefficients of the fitted `LogisticRegression`, on standardised features, "
    "so they are comparable to each other. Positive pushes towards churn.",
    "",
    "| Feature | Coefficient |",
    "| :--- | ---: |",
]
for k, v in feature_importance.sort_values(key=abs, ascending=False).items():
    resumen.append(f"| {k} | {v:+.4f} |")
resumen += [
    "",
    "## Visualisations",
    "",
    "The confusion matrix is also saved as `confusion_matrix.png`.",
    "",
]

with open('analysis_results.md', 'w', encoding='utf-8') as f:
    f.write("\n".join(resumen))

print("Write-up saved to analysis_results.md")
print("Analysis complete.")
