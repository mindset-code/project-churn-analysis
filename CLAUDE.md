# Predictive Churn Analysis — Data Science / BI

## Qué es este proyecto
Análisis predictivo de fuga de clientes (Churn) usando **Regresión Logística**. Identifica los factores que predicen qué clientes abandonarán el servicio, permitiendo implementar estrategias de retención proactivas.

Proyecto de portafolio LinkedIn — conecta experiencia en SalesOps/RevOps con Data Science aplicado a negocio.

## Stack técnico
- **Python:** Pandas, Scikit-learn, Matplotlib, Seaborn
- **Modelo:** Logistic Regression (clasificación binaria)
- **Evaluación:** Confusion Matrix, Accuracy, Precision, Recall

## Archivos clave
- `churn_analysis.py` — preprocesamiento, entrenamiento del modelo y evaluación
- `churn_data.csv` — dataset sintético de clientes (features de comportamiento y contrato)
- `generate_data.py` — generador del dataset sintético
- `confusion_matrix.png` — visualización de la matriz de confusión
- `analysis_results.md` — informe de hallazgos y métricas del modelo

## Pipeline del análisis

```
churn_data.csv
    → Preprocesamiento (encoding, train/test split)
    → Entrenamiento Logistic Regression
    → Evaluación (Confusion Matrix, métricas)
    → analysis_results.md + confusion_matrix.png
```

## Hallazgos clave
Los predictores más fuertes de churn identificados:
- **Duración corta del contrato** — clientes con contratos mensuales churnan más
- **Alto número de tickets de soporte** — señal de insatisfacción
- Correlación negativa entre antigüedad del cliente y probabilidad de churn

## Cómo ejecutar
```bash
pip install pandas scikit-learn matplotlib seaborn
python churn_analysis.py
```

## Extensiones posibles
- Probar modelos más potentes: Random Forest, XGBoost
- Añadir SHAP values para explicabilidad del modelo
- Crear dashboard de monitoreo de churn en tiempo real con Streamlit
- Conectar con CRM real para scoring automático de clientes en riesgo
- Automatizar alertas via n8n cuando un cliente supera el umbral de riesgo

## Relevancia para el portafolio
Demuestra: Machine Learning aplicado a negocio, Python/Scikit-learn, RevOps, customer analytics.
Roles objetivo: Data Scientist, BI Analyst, RevOps Analyst, Analytics Engineer.
