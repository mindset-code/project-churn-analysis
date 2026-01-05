import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('churn_data.csv')

# 2. Data Preprocessing
# Convert categorical features to dummy variables
df = pd.get_dummies(df, columns=['SubscriptionType'], drop_first=True)

# Define features (X) and target (y)
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model Training (Logistic Regression for interpretability)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# 5. Feature Importance Analysis (Coefficients)
feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)

# 6. Visualization (Confusion Matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix for Churn Prediction')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# 7. Save Key Findings to a Markdown file
with open('analysis_results.md', 'w') as f:
    f.write("# Churn Prediction Analysis - Key Findings\n\n")
    f.write("## Model Performance (Logistic Regression)\n")
    f.write(f"Accuracy: {report['accuracy']:.2f}\n\n")
    f.write("### Classification Report\n")
    f.write(pd.DataFrame(report).transpose().to_markdown())
    f.write("\n\n## Feature Importance (Model Coefficients)\n")
    f.write("The following features were the most significant predictors of customer churn:\n")
    f.write(feature_importance.head(5).to_markdown())
    f.write("\n\n## Visualizations\n")
    f.write("A Confusion Matrix visualization has been saved as `confusion_matrix.png`.\n")

print("Churn analysis complete. Results saved to analysis_results.md and confusion_matrix.png.")
