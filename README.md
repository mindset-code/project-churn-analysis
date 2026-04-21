# Predictive Churn Analysis — Data Science Portfolio

> **Data Science Portfolio Project** · Logistic Regression · 81.4% Accuracy · AUC 0.881
> **Status:** Finished · Deployed to production (2026-04)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-%E2%86%92%20Open%20Dashboard-a78bfa?style=for-the-badge&logo=firebase&logoColor=white)](https://proyectos-personales.web.app/churn)
[![Portfolio](https://img.shields.io/badge/Portfolio-proyectos--personales.web.app-60a5fa?style=for-the-badge)](https://proyectos-personales.web.app)

---

## Project Status

| Phase | Status |
|---|---|
| Data generation (synthetic dataset) | Done |
| Model training & tuning | Done |
| Evaluation (AUC 0.881, 81.4% accuracy) | Done |
| Business findings writeup | Done |
| React dashboard deployment | Done |

**Current phase:** maintenance — model live on portfolio.

---

## What This Project Does

Builds a Logistic Regression model to predict customer churn and identify the key business drivers behind it. The analysis connects Data Science directly to Revenue Operations — enabling proactive retention strategies before customers leave.

**Live dashboard → [proyectos-personales.web.app/churn](https://proyectos-personales.web.app/churn)**

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.4% |
| **AUC-ROC** | 0.881 |
| **Precision** | ~78% |
| **Recall** | ~76% |

The model correctly identifies over 3 out of 4 customers at risk of churning, giving retention teams actionable lead time.

---

## Key Findings

- **Short contract duration** is the strongest predictor of churn — customers on month-to-month contracts churn at 3× the rate of annual subscribers
- **High support ticket volume** correlates strongly with imminent churn — a signal of unresolved friction
- **Low product engagement** (logins, feature usage) precedes churn by 30–60 days on average
- Combining these three signals captures ~70% of churners before they cancel

---

## Dataset

Synthetic dataset of 1,000 customers with realistic churn dynamics:

```
churn_data.csv   1,000 rows  · 10 features  · binary churn label
```

| Feature | Type | Description |
|---------|------|-------------|
| `contract_duration` | Numeric | Months on subscription |
| `support_tickets` | Numeric | Support contacts last 90 days |
| `monthly_spend` | Numeric | Average monthly revenue |
| `logins_per_month` | Numeric | Product engagement |
| `plan_type` | Categorical | Basic / Pro / Enterprise |
| `churned` | Binary | Target variable (0 = retained, 1 = churned) |

---

## Skills Demonstrated

- **Machine Learning:** Logistic Regression with Scikit-learn, hyperparameter tuning
- **Data Preprocessing:** Encoding, train/test split, feature scaling
- **Model Evaluation:** Confusion matrix, ROC curve, AUC, precision/recall
- **Business Translation:** Converting model outputs into actionable retention strategies
- **Python / Pandas:** End-to-end ML pipeline from raw data to scored predictions
- **React Dashboard:** Interactive visualization of model results deployed on Firebase

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | Python 3.12, Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web | React 19, Vite, Recharts — [proyectos-personales.web.app/churn](https://proyectos-personales.web.app/churn) |
| Hosting | Firebase Hosting (Spark plan) |

---

## How to Run

```bash
# Clone
git clone https://github.com/mindset-code/project-churn-analysis.git
cd project-churn-analysis

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# Run analysis
python churn_analysis.py
# Output: confusion_matrix.png + printed model metrics
```

---

## Repository Structure

```
project-churn-analysis/
├── churn_analysis.py       # Main ML pipeline: preprocess → train → evaluate
├── churn_data.csv          # Synthetic customer dataset (1,000 rows)
├── confusion_matrix.png    # Model evaluation visualization
└── analysis_results.md     # Detailed findings and business recommendations
```

---

## Links

- **Portfolio:** [proyectos-personales.web.app](https://proyectos-personales.web.app)
- **LinkedIn:** [Mindset & Code](https://www.linkedin.com/company/mindset-code)
- **Email:** contacto@mindset-code.com

---

*Built by [Mindset & Code](https://github.com/mindset-code) · Data & BI Analyst · MBA · ISC2 CC*
