---

# Machine Learning Projects

This repository contains several machine learning projects covering **regression**, **classification**, and **clustering** tasks. Each project includes data preprocessing, model training, evaluation, and visualization.

---

## Projects Overview

### Task 1: Student Score Prediction

* **Objective:** Predict student exam scores based on hours studied using linear regression.
* **Dataset:** `StudentPerformanceFactors.csv`
* **Key Steps:**

  * Load and explore dataset
  * Split into training and testing sets
  * Train a **Linear Regression** model
  * Evaluate using **Mean Squared Error (MSE)** and **R² Score**
  * Visualize actual vs predicted scores

---

### Task 2: Customer Segmentation

* **Objective:** Segment mall customers into groups based on demographics and spending behavior.
* **Dataset:** `Mall_Customers.csv`
* **Key Steps:**

  * Load and preprocess dataset
  * Convert categorical features (Gender) to numeric
  * Use **KMeans clustering** to segment customers
  * Determine optimal clusters with **Elbow Method**
  * Visualize clusters (Annual Income vs Spending Score)
  * Save clustered dataset to `Mall_Customers_Clustered.csv`

---

### Task 3: Forest Cover Type Classification

* **Objective:** Predict forest cover type based on cartographic features.
* **Dataset:** `covtype.csv`
* **Key Steps:**

  * Load and explore dataset
  * Split into training and testing sets
  * Train a **Random Forest Classifier**
  * Evaluate using **classification report** and **confusion matrix**
  * Compute and visualize **feature importances**
  * Plot confusion matrix heatmap

---

### Task 4: Loan Approval Prediction

* **Objective:** Predict whether a loan application will be approved.
* **Dataset:** `loan_approval_dataset.csv`
* **Key Steps:**

  * Load and preprocess dataset
  * Handle missing values and encode categorical features
  * Split into training and testing sets
  * Handle class imbalance using **SMOTE**
  * Scale numerical features using **StandardScaler**
  * Train **Logistic Regression** and **Decision Tree Classifier**
  * Evaluate using **precision, recall, F1-score**, and confusion matrices
  * Visualize loan status distribution, feature correlations, and CIBIL score trends

---

## Installation

1. Clone the repository:

```
git clone <repository-url>
cd <repository-folder>
```

2. Install required Python packages:

```
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Usage

1. Place the dataset CSV files in the repository root.
2. Open the corresponding Python script for the task.
3. Run the script:

```
python Task1_StudentScorePrediction.py
python Task2_CustomerSegmentation.py
python Task3_ForestCoverTypeClassification.py
python Task4_LoanApprovalPrediction.py
```

4. Visualizations will be displayed during execution.

---

## Notes

* Ensure Python **3.8+** is installed.
* SMOTE is applied **only to training data** in Task 4 to prevent data leakage.
* Target variables in classification tasks are encoded to numeric values before evaluation.

---

## License

This repository is for **educational purposes**. You can use it freely for learning and experimentation.

---