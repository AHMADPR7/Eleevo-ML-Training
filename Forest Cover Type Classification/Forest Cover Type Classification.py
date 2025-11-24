# Author : Ahmed Ezzat Abd El-Monem
# Task 3: Forest Cover Type Classification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

base = os.getcwd() 
full_path = os.path.join(base, "Forest Cover Type Classification", "covtype.csv")

# Load the dataset
df = pd.read_csv(full_path)
print(df.head())
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False) 
print("\nFeature Importances:")
print(feature_importance_df)
# Visualize feature importances
top_features = feature_importance_df.head(10)

plt.figure(figsize=(8,6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances in Forest Cover Type Classification')
plt.gca().invert_yaxis()  # Most important at the top
plt.tight_layout()
plt.show()

#Visualize confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Forest Cover Type Classification')
plt.show()

