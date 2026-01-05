# Churn Prediction Analysis - Key Findings

## Model Performance (Logistic Regression)
Accuracy: 0.68

### Classification Report
|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.683168 | 0.811765 |   0.741935 |    170    |
| 1            |    0.673469 | 0.507692 |   0.578947 |    130    |
| accuracy     |    0.68     | 0.68     |   0.68     |      0.68 |
| macro avg    |    0.678319 | 0.659729 |   0.660441 |    300    |
| weighted avg |    0.678965 | 0.68     |   0.671307 |    300    |

## Feature Importance (Model Coefficients)
The following features were the most significant predictors of customer churn:
|                         |            0 |
|:------------------------|-------------:|
| SupportTickets          |  0.222845    |
| MonthlyCharges          |  0.00898457  |
| Age                     |  0.00273665  |
| TotalUsageHours         |  0.000736328 |
| ContractDuration_Months | -0.0225455   |

## Visualizations
A Confusion Matrix visualization has been saved as `confusion_matrix.png`.
