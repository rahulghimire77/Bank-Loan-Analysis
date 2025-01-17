Loan Default Prediction
This project demonstrates a machine learning pipeline for predicting the grade of a loan (which can be an indicator of loan default risk) using various classification algorithms. It includes:

Data preprocessing and feature engineering
Model training using different classifiers (Random Forest, Gradient Boosting, Logistic Regression, SVC, XGBoost, KNN)
Hyperparameter tuning for Random Forest and XGBoost
Evaluation using performance metrics such as accuracy score and classification report
Visualization of the confusion matrix for the best model
Table of Contents
Data
Project Structure
Prerequisites
Setup and Usage
Model Training and Evaluation
Results
License
Data
Source: The code expects a CSV file named train.csv located at:

bash
Copy
C:/Users/rgh/Documents/deep learning/Datathon/techparva3-datathon/train.csv
Adjust the path to the location of your dataset if needed.

Description: The dataset contains information about loans (e.g., annual income, debt-to-income ratio, interest rates, loan amount, home ownership, etc.) as well as a target label (grade) indicating the credit grade of the loan.

Project Structure
bash
Copy
.
├── README.md                    # Documentation
├── loan_default_prediction.py   # The main Python script (or notebook)
└── requirements.txt             # Python libraries required (Optional)
(Adjust the file names/structure as needed.)

Prerequisites
Make sure you have Python 3.7+ installed on your system. Additionally, the following libraries are used in the project:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
You can install the required packages using pip:

bash
Copy
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
(Optional) If you have a requirements.txt file, simply run:

bash
Copy
pip install -r requirements.txt
Setup and Usage
Clone or download the repository:

bash
Copy
git clone https://github.com/YourUsername/Loan-Default-Prediction.git
cd Loan-Default-Prediction
Place your dataset (train.csv) in the correct path. Update the df = pd.read_csv("C:/Users/rgh/Documents/deep learning/Datathon/techparva3-datathon/train.csv") line in the script if your data resides elsewhere.

Run the script (assuming you’ve named it loan_default_prediction.py):

bash
Copy
python loan_default_prediction.py
Results (accuracy, classification reports, and confusion matrix) will be displayed in the console or as plots.

Model Training and Evaluation
Data Preprocessing
Drop redundant or unnecessary columns:
Copy
application_type, emp_title, issue_date_year, issue_date_hour, ...
Convert string features with numerical values to float (e.g., emp_length, term).
Convert date columns to datetime objects and extract relevant features like month, day, or weekday.
Handle missing values using SimpleImputer (mean/median imputation) or KNNImputer.
Scale numerical features with RobustScaler.
Encoding and Transformation
Categorical features are transformed using OneHotEncoder (with drop='first' to avoid dummy variable trap).
Target labels (grade) are label-encoded via LabelEncoder.
Training Classifiers
Random Forest
Gradient Boosting
Support Vector Classifier (SVC)
Logistic Regression
XGBoost
K-Nearest Neighbors (KNN)
Each model is fit on the training set, and predictions are made on the test set.

Hyperparameter Tuning
Random Forest: RandomizedSearchCV with a parameter grid (n_estimators, max_features, max_depth, max_samples).
XGBoost: RandomizedSearchCV with a parameter grid (learning_rate, max_depth, min_child_weight, gamma, colsample_bytree).
Evaluation
Accuracy Score: accuracy_score(y_test, y_pred)
Classification Report: classification_report(y_test, y_pred) for precision, recall, and F1-score.
Confusion Matrix: Visualized using seaborn.heatmap.
Results
Various models yield different accuracy scores and classification metrics.
The confusion matrix of the best model (XGBoost after tuning or Random Forest with the best hyperparameters) is displayed for detailed insight.
Example:

python
Copy
Best XGB Accuracy: 0.85
Classification Report:
              precision    recall  f1-score   support
...
(Values will differ based on your dataset and random states.)

License
This project is open-source. You are free to use, modify, and distribute the code. If you use it in your own project or research, a citation or link back to this repository would be appreciated.

Feel free to customize this README with additional details about your findings, data exploration insights, or any domain-specific knowledge about loan grading or default prediction.
