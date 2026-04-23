# Banking Customer Churn Prediction
## 1. Problem & User
This project aims to predict which banking customers are likely to churn (close their accounts) so that the bank can proactively implement targeted retention strategies. The primary users are bank marketing and customer retention teams who need to identify at-risk customers before they leave, reducing revenue loss and customer acquisition costs.
## 2. Data
- **Source:** https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/data
- **Access date:** 2026-04-19  
- **Key fields:** | Field | Description |
|-------|-------------|
| RowNumber | Sequential number assigned to each row |
| CustomerId | Unique identifier for each customer |
| Surname | Customer's surname |
| CreditScore | Credit score of the customer |
| Geography | Customer's geographical location (e.g., country or region) |
| Gender | Gender of the customer |
| Age | Age of the customer |
| Tenure | Number of years the customer has been with the bank |
| Balance | Account balance of the customer |
| NumOfProducts | Number of bank products the customer has |
| HasCrCard | Whether the customer has a credit card (yes/no) |
| IsActiveMember | Whether the customer is an active member (yes/no) |
| EstimatedSalary | Estimated salary of the customer |
| Exited | Whether the customer has exited the bank (yes/no) |
## 3. Methods
- The analysis follows a standard machine learning pipeline:
- Data Loading & Initial Checks – No missing values, no duplicates, class imbalance noted.
- Exploratory Data Analysis (EDA) – Visualized churn distribution, geography, age, and balance patterns.
- Preprocessing – Label encoding for Geography (France/Spain/Germany) and Gender (Female/Male), standardization of numerical features.
- Train/Test Split – 70/30 split with stratification to preserve churn ratio.
- Model Training – Logistic Regression (baseline) and Decision Tree (max_depth=5).
- Evaluation Metrics – Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix.
## 4. Key Findings
- Decision Tree outperformed Logistic Regression – F1-Score: 0.545 vs 0.304, ROC AUC: 0.852 vs 0.787.
- Age is the most important predictor – Older customers (45–60) show significantly higher churn risk.
- German customers have the highest churn rate (~32% vs 16% in France/Spain).
- Active members are less likely to churn – IsActiveMember has strong negative coefficient in logistic regression.
- Recall remains low for both models – Identifying churned customers is challenging due to class imbalance (only 20.4% churn).
## 5. How to Run
bash
# Requirements
pip install pandas numpy matplotlib scikit-learn
# Run the notebook
jupyter notebook "Banking Customer Churn Prediction.ipynb"
Make sure Churn_Modelling.csv is in the same directory as the notebook.
