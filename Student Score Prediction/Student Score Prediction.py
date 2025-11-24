# Author : Ahmed Ezzat Abd ElMonem
# Task 1: Student Score Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

base = os.getcwd() 
full_path = os.path.join(base, "Student Score Prediction", "StudentPerformanceFactors.csv")
# Load the dataset
df = pd.read_csv(full_path)
print(df.head())
X = df[['Hours_Studied']]
y = df['Exam_Score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')
plt.scatter(X_test, y_pred, color='red', label='Predicted Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Student Score Prediction')
plt.legend()
plt.show()


