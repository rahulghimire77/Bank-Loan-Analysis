# Loan Default Prediction (Datathon Project)

This project demonstrates a machine learning pipeline for predicting the **grade** of a loan (which can be an indicator of loan default risk) using various classification algorithms. It includes:

- Data preprocessing and feature engineering  
- Model training using different classifiers (Random Forest, Gradient Boosting, Logistic Regression, SVC, XGBoost, KNN)  
- Hyperparameter tuning for Random Forest and XGBoost  
- Evaluation using performance metrics such as accuracy score and classification report  
- Visualization of the confusion matrix for the best model  

## Table of Contents

- [Data](#data)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)

---

## Data

- **Source**: The code expects a CSV file named `train.csv`

- **Description**: The dataset contains information about loans (e.g., annual income, debt-to-income ratio, interest rates, loan amount, home ownership, etc.) as well as a target label (`grade`) indicating the credit grade of the loan.

---
## Model Training and Evaluation

In this phase, we prepare the data, transform it, train various classifiers, tune hyperparameters, and evaluate performance metrics. Below is a concise overview of the steps—combined into a single section—that you can copy into your notebook.

**Data Preprocessing**:  
1. Drop redundant or unnecessary columns (e.g., `application_type`, `emp_title`, `issue_date_year`, `issue_date_hour`).  
2. Convert string columns with numerical values to floats (e.g., `emp_length`, `term`).  
3. Convert date columns to `datetime` objects and extract relevant features (e.g., `month`, `day`, `weekday`).  
4. Handle missing values using `SimpleImputer` (mean or median) or `KNNImputer`.  
5. Scale numerical features with `RobustScaler`.  

**Encoding and Transformation**:  
- Transform categorical features using `OneHotEncoder` (with `drop='first'` to avoid dummy variable trap).  
- Label-encode the target variable (`grade`) using `LabelEncoder`.  

**Training Classifiers**:  
We train multiple models on the processed dataset:
1. Random Forest  
2. Gradient Boosting  
3. Support Vector Classifier (SVC)  
4. Logistic Regression  
5. XGBoost  
6. K-Nearest Neighbors (KNN)  

Each classifier is fit on the training set, and predictions are made on the test set.

**Hyperparameter Tuning**:  
- Random Forest: Use `RandomizedSearchCV` with a grid including parameters such as `n_estimators`, `max_features`, `max_depth`, `max_samples`.  
- XGBoost: Use `RandomizedSearchCV` with parameters such as `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `colsample_bytree`.  

**Evaluation**:  
- **Accuracy Score**: `accuracy_score(y_test, y_pred)`  
- **Classification Report**: `classification_report(y_test, y_pred)` for precision, recall, and F1-scores  
- **Confusion Matrix**: Visualize with `seaborn.heatmap` to understand misclassifications  

By comparing accuracy and other metrics (precision, recall, F1-score) across these models, we identify the best-performing classifier. The confusion matrix of the tuned best model (often XGBoost or Random Forest) provides deeper insights into predictions versus actual outcomes.

## Results

After training and tuning each of the models (Random Forest, Gradient Boosting, Logistic Regression, SVC, XGBoost, and KNN), you can compare their performance using various metrics.

- **Accuracy Scores**: 
  - For each model, compute `accuracy_score(y_test, y_pred)` to get the percentage of correct predictions.
- **Classification Reports**: 
  - Use `classification_report(y_test, y_pred)` to evaluate precision, recall, and F1-score for each class.
- **Confusion Matrix**: 
  - Visualize the confusion matrix with `seaborn.heatmap` to understand the distribution of predictions vs. actual classes. This helps identify which classes are often misclassified.

Below is an example of the output you might see for a tuned XGBoost model:

Best XGB Accuracy: 0.85

...


