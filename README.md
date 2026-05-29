# Predictive Churn Analysis — Customer Retention ML

> **Data Science portfolio project** · Python · scikit-learn · Logistic Regression
> **Status:** Finished · Live on portfolio
> An end-to-end churn-prediction pipeline — from synthetic data to a trained, evaluated model and an interactive dashboard — that identifies which customers are about to leave and *why*.

> 🇬🇧 **English version first.** · 🇪🇸 **La versión en español está más abajo** → [ir a Español](#-español).

[![Live Demo](https://img.shields.io/badge/Live%20Demo-%E2%86%92%20Open%20Dashboard-a78bfa?style=for-the-badge&logo=firebase&logoColor=white)](https://proyectos-personales.web.app/churn)
[![Portfolio](https://img.shields.io/badge/Portfolio-proyectos--personales.web.app-60a5fa?style=for-the-badge&logo=firebase&logoColor=white)](https://proyectos-personales.web.app)
[![Stack](https://img.shields.io/badge/Stack-Python%20%C2%B7%20scikit--learn-3776AB?style=for-the-badge&logo=scikitlearn&logoColor=white)](.)
[![Domain](https://img.shields.io/badge/Domain-Data%20Science%20%C2%B7%20Retention-16a34a?style=for-the-badge)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## The problem this solves

Acquiring a customer costs far more than keeping one — so the most valuable question a subscription business can answer is *"who is about to churn, and what's driving it?"* This project builds the machine-learning pipeline that answers it: it predicts churn probability per customer and surfaces the **business drivers** behind the prediction, giving retention teams actionable lead time instead of a post-mortem.

It demonstrates the full data-science workflow — data generation, preprocessing, handling class imbalance, model training, rigorous evaluation, and communicating results — connecting Data Science directly to Revenue Operations.

**▶ Live dashboard: [proyectos-personales.web.app/churn](https://proyectos-personales.web.app/churn)**

---

## Model performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.4% |
| **AUC-ROC** | 0.881 |
| **Precision (churn)** | ~0.78 |
| **Recall (churn)** | ~0.76 |

The model correctly flags more than 3 of every 4 at-risk customers — enough lead time to act before they cancel. AUC-ROC (0.881) is the headline metric because the classes are imbalanced.

---

## How it works

1. **Data** — `generate_data.py` builds a synthetic, labelled customer dataset.
2. **Preprocessing** — one-hot encoding of `SubscriptionType`, feature selection, `StandardScaler` normalization.
3. **Class imbalance** — `class_weight='balanced'` + a stratified train/test split so the minority (churned) class isn't drowned out.
4. **Model** — `LogisticRegression` (`max_iter=1000`, `random_state=42`) — chosen for interpretability: coefficients map directly to business drivers.
5. **Evaluation** — accuracy, precision/recall, ROC-AUC and a confusion matrix (`confusion_matrix.png`).

**Features:** Age · MonthlyCharges · TotalUsageHours · SupportTickets · ContractDuration · TenureMonths · NumProducts · SubscriptionType.

---

## Key findings

- **Short contract duration** is the strongest churn predictor — month-to-month customers churn well above annual subscribers.
- **High support-ticket volume** signals unresolved friction and imminent churn.
- **Low product engagement** (usage hours) precedes churn — the earliest warning sign.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | scikit-learn (LogisticRegression, StandardScaler) |
| Data | Pandas · NumPy |
| Visualization | Matplotlib · Seaborn · React (dashboard) |

---

## Getting started

```bash
git clone https://github.com/mindset-code/project-churn-analysis.git
cd project-churn-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py     # build the synthetic dataset
python churn_analysis.py    # train, evaluate, export metrics + confusion_matrix.png
```

---

## Repository structure

```
project-churn-analysis/
├── churn_analysis.py     # full pipeline: preprocessing → model → metrics
├── generate_data.py      # synthetic dataset generator
├── confusion_matrix.png  # model evaluation output
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT
└── README.md
```

---

## License & contact

Released under the **[MIT License](LICENSE)**.

- **Portfolio:** [proyectos-personales.web.app](https://proyectos-personales.web.app)
- **LinkedIn:** [Mindset & Code](https://www.linkedin.com/company/mindset-code)
- **Email:** contacto@mindset-code.com

---

# 🇪🇸 Español

# Predictive Churn Analysis — ML de retención de clientes

> **Proyecto de portafolio de Data Science** · Python · scikit-learn · Regresión Logística
> **Estado:** Terminado · Publicado en el portafolio
> Pipeline de predicción de churn de extremo a extremo —de los datos sintéticos a un modelo entrenado y evaluado y un dashboard interactivo— que identifica qué clientes están a punto de irse y *por qué*.

> 🇪🇸 Traducción al español. La versión en inglés está al inicio → [ir a English](#predictive-churn-analysis--customer-retention-ml).

---

## El problema que resuelve

Captar un cliente cuesta mucho más que retenerlo — así que la pregunta más valiosa para un negocio de suscripción es *"¿quién está a punto de irse y qué lo provoca?"*. Este proyecto construye el pipeline de machine learning que la responde: predice la probabilidad de churn por cliente y expone los **drivers de negocio** detrás de la predicción, dando a los equipos de retención tiempo para actuar en lugar de una autopsia.

Demuestra el flujo completo de data science —generación de datos, preprocesamiento, manejo de clases desbalanceadas, entrenamiento, evaluación rigurosa y comunicación de resultados— conectando Data Science directamente con Revenue Operations.

**▶ Dashboard en vivo: [proyectos-personales.web.app/churn](https://proyectos-personales.web.app/churn)**

---

## Rendimiento del modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 81,4% |
| **AUC-ROC** | 0,881 |
| **Precisión (churn)** | ~0,78 |
| **Recall (churn)** | ~0,76 |

El modelo detecta más de 3 de cada 4 clientes en riesgo — margen suficiente para actuar antes de la cancelación. El AUC-ROC (0,881) es la métrica principal porque las clases están desbalanceadas.

---

## Cómo funciona

1. **Datos** — `generate_data.py` genera un dataset sintético etiquetado de clientes.
2. **Preprocesamiento** — one-hot encoding de `SubscriptionType`, selección de features, normalización con `StandardScaler`.
3. **Desbalanceo** — `class_weight='balanced'` + split estratificado para que la clase minoritaria (churned) no se diluya.
4. **Modelo** — `LogisticRegression` (`max_iter=1000`, `random_state=42`) — elegido por interpretabilidad: los coeficientes mapean directamente a drivers de negocio.
5. **Evaluación** — accuracy, precisión/recall, ROC-AUC y matriz de confusión (`confusion_matrix.png`).

**Features:** Age · MonthlyCharges · TotalUsageHours · SupportTickets · ContractDuration · TenureMonths · NumProducts · SubscriptionType.

---

## Hallazgos clave

- **La duración corta de contrato** es el predictor más fuerte — los clientes mes a mes churnan muy por encima de los anuales.
- **El alto volumen de tickets de soporte** señala fricción no resuelta y churn inminente.
- **El bajo engagement** (horas de uso) precede al churn — la señal de alerta más temprana.

---

## Stack técnico

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.12 |
| ML | scikit-learn (LogisticRegression, StandardScaler) |
| Datos | Pandas · NumPy |
| Visualización | Matplotlib · Seaborn · React (dashboard) |

---

## Cómo empezar

```bash
git clone https://github.com/mindset-code/project-churn-analysis.git
cd project-churn-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py     # genera el dataset sintético
python churn_analysis.py    # entrena, evalúa y exporta métricas + confusion_matrix.png
```

---

## Estructura del repositorio

```
project-churn-analysis/
├── churn_analysis.py     # pipeline completo: preprocesamiento → modelo → métricas
├── generate_data.py      # generador de dataset sintético
├── confusion_matrix.png  # salida de evaluación del modelo
├── requirements.txt      # dependencias Python
├── LICENSE               # MIT
└── README.md
```

---

## Licencia y contacto

Publicado bajo la **[Licencia MIT](LICENSE)**.

- **Portafolio:** [proyectos-personales.web.app](https://proyectos-personales.web.app)
- **LinkedIn:** [Mindset & Code](https://www.linkedin.com/company/mindset-code)
- **Email:** contacto@mindset-code.com

---

*Built by [Mindset & Code](https://github.com/mindset-code) · Data & BI Analyst · MBA · ISC2 CC*
