# Task 4: Loan Approval Prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import os

base = os.getcwd() 
full_path = os.path.join(base, "Loan Approval Prediction", "loan_approval_dataset.csv")

# Load the dataset
df = pd.read_csv(full_path)
print("Dataset shape:", df.shape)
print(df.head())

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Missing Values Handling
df.fillna({
    'no_of_dependents': df['no_of_dependents'].mode()[0],
    'education': df['education'].mode()[0],
    'self_employed': df['self_employed'].mode()[0],
    'cibil_score': df['cibil_score'].mean()
}, inplace=True)
df.dropna(inplace=True)

# ============================
# Encode Categorical Variables
# ============================
label_encoders = {}
categorical_features = ['education', 'self_employed']

for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Encode target variable
target_le = LabelEncoder()
df['loan_status'] = target_le.fit_transform(df['loan_status'])  # Approved=1, Rejected=0
y = df['loan_status']

# ============================
# Prepare Features
# ============================
X = df.drop(columns=['loan_id', 'loan_status'])
numerical_features = [
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

# ============================
# Train-Test Split BEFORE SMOTE
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# Apply SMOTE Only to Training Data
# ============================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Resampled shape:", X_train.shape, y_train.shape)

# ============================
# Scale Numerical Features
# ============================
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# ============================
# Logistic Regression
# ============================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

# ============================
# Decision Tree
# ============================
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_dec_tree = dec_tree.predict(X_test)

print("\n=== Decision Tree Report ===")
print(classification_report(y_test, y_pred_dec_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dec_tree))

# ============================
# Visualizations
# ============================

# Loan Status Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution')
plt.show()

# Feature Correlation Heatmap (numeric only)
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# CIBIL Score vs Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='cibil_score', data=df)
plt.title('CIBIL Score vs Loan Status')
plt.show()
