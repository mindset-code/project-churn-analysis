"""
Synthetic Churn Dataset Generator — improved correlations & realism.
Run: python generate_data.py
Output: churn_data.csv
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 2000  # 2x more customers

age                  = np.random.randint(22, 68, N)
subscription         = np.random.choice(['Basic', 'Standard', 'Premium'], N, p=[0.40, 0.35, 0.25])
monthly_charges      = np.where(
    subscription == 'Basic',    np.random.uniform(20,  60,  N),
    np.where(subscription == 'Standard', np.random.uniform(55, 100, N),
                                          np.random.uniform(95, 150, N))
).round(2)
total_usage          = np.random.uniform(30, 600, N).round(1)
support_tickets      = np.random.poisson(2.2, N).clip(0, 12)
contract_months      = np.random.choice([1, 12, 24], N, p=[0.45, 0.32, 0.23])
tenure_months        = np.random.randint(1, 60, N)
num_products         = np.random.choice([1, 2, 3, 4], N, p=[0.40, 0.30, 0.20, 0.10])

# Base churn probability driven by multiple factors
churn_prob = (
    0.10
    + 0.22 * (support_tickets >= 5).astype(float)
    + 0.18 * (monthly_charges > 100).astype(float)
    + 0.15 * (contract_months == 1).astype(float)
    - 0.12 * (subscription == 'Premium').astype(float)
    - 0.10 * (tenure_months > 24).astype(float)
    - 0.08 * (num_products >= 3).astype(float)
    + 0.10 * (total_usage < 80).astype(float)
    - 0.06 * (contract_months == 24).astype(float)
).clip(0.02, 0.92)

churn = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

df = pd.DataFrame({
    'CustomerID':           range(1, N + 1),
    'Age':                  age,
    'SubscriptionType':     subscription,
    'MonthlyCharges':       monthly_charges,
    'TotalUsageHours':      total_usage,
    'SupportTickets':       support_tickets,
    'ContractDuration_Months': contract_months,
    'TenureMonths':         tenure_months,
    'NumProducts':          num_products,
    'Churn':                churn,
})

df.to_csv('churn_data.csv', index=False)
print(f"Dataset generated: {N} customers, {churn.mean():.1%} churn rate")
print(df['Churn'].value_counts().to_string())
