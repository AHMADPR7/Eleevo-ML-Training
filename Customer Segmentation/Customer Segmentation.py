# Author : Ahmed Ezzat Abd ElMonem
# Task 2: Customer Segmentation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

base = os.getcwd() 
full_path = os.path.join(base, "Customer Segmentation", "Mall_Customers.csv")
# Load the dataset
df = pd.read_csv(full_path)

print(df.head())
X = df[['Genre','Age','Annual Income (k$)', 'Spending Score (1-100)']]
# Convert categorical 'Genre' to numerical
X['Genre'] = X['Genre'].map({'Male': 0, 'Female': 1})
# Determine the optimal number of clusters using the Elbow Method
wcss = []   
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with the optimal number of clusters  k=3 to 5
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)
# Use only 2 columns for visualization
x_col = 'Annual Income (k$)'
y_col = 'Spending Score (1-100)'

plt.figure(figsize=(8,6))

for i in range(k):
    plt.scatter(
        X.loc[y_kmeans == i, x_col],
        X.loc[y_kmeans == i, y_col],
        s=100,
        label=f'Cluster {i+1}'
    )

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, X.columns.get_loc(x_col)],
    kmeans.cluster_centers_[:, X.columns.get_loc(y_col)],
    s=300,
    c='yellow',
    edgecolor='black',
    label='Centroids'
)

plt.title('Customer Segmentation (Income vs Spending)')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend()
plt.show()
# Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans
# Save the clustered data to a new CSV file
df.to_csv('Mall_Customers_Clustered.csv', index=False)

