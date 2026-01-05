import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 1000

# Generate synthetic data
data = {
    'CustomerID': range(1, n_customers + 1),
    'Age': np.random.randint(25, 65, n_customers),
    'SubscriptionType': np.random.choice(['Basic', 'Standard', 'Premium'], n_customers, p=[0.4, 0.35, 0.25]),
    'MonthlyCharges': np.random.uniform(20, 150, n_customers).round(2),
    'TotalUsageHours': np.random.uniform(50, 500, n_customers).round(2),
    'SupportTickets': np.random.randint(0, 10, n_customers),
    'ContractDuration_Months': np.random.choice([1, 12, 24], n_customers, p=[0.5, 0.3, 0.2]),
    'Churn': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]) # 80% No Churn, 20% Churn
}

df = pd.DataFrame(data)

# Introduce correlation: Higher charges, lower contract duration, and more tickets lead to higher churn
df.loc[(df['MonthlyCharges'] > 100) & (df['ContractDuration_Months'] == 1), 'Churn'] = np.random.choice([0, 1], len(df[(df['MonthlyCharges'] > 100) & (df['ContractDuration_Months'] == 1)]), p=[0.3, 0.7])
df.loc[df['SupportTickets'] > 7, 'Churn'] = np.random.choice([0, 1], len(df[df['SupportTickets'] > 7]), p=[0.2, 0.8])

# Save the dataset
df.to_csv('/home/ubuntu/project1_churn/churn_data.csv', index=False)

print("Synthetic Churn Data generated successfully.")
