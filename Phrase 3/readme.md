# Phase 3: Intermediate Machine Learning and Pipelines

This document outlines the key topics, problems, and mini-projects to deepen your understanding of intermediate machine learning techniques, including feature engineering, model tuning, and creating automated ML pipelines. Completing these tasks will strengthen your ability to handle real-world ML problems.

## Topics and Problems to Solve

### 1. Feature Engineering
#### Key Concepts:
- Handling missing data using advanced imputation techniques (e.g., KNN Imputer).
- Encoding techniques: One-Hot Encoding, Label Encoding, Target Encoding.
- Feature scaling: Standardization and Normalization.
- Feature creation: Polynomial features, interaction terms.

#### Problems:
1. Handle missing values in a dataset using the KNN imputer.
2. Encode categorical variables using One-Hot Encoding and Target Encoding.
3. Normalize a dataset using MinMaxScaler and StandardScaler.
4. Generate polynomial features for a regression problem and evaluate model performance.
5. Create new features by combining existing ones (e.g., `price_per_unit = price / quantity`).

---

### 2. Model Evaluation and Diagnostics
#### Key Concepts:
- Performance metrics for classification: Precision, Recall, F1-score, ROC-AUC.
- Performance metrics for regression: RMSE, MAE, R².
- Analyzing residuals to evaluate regression models.
- Creating confusion matrices and ROC curves for classification models.

#### Problems:
1. Evaluate a regression model using RMSE, MAE, and R².
2. Plot residuals and analyze their patterns.
3. Create and interpret a confusion matrix for a classification model.
4. Generate and analyze an ROC curve for a binary classification problem.
5. Implement k-fold cross-validation and compare model performance.

---

### 3. Hyperparameter Tuning
#### Key Concepts:
- Importance of hyperparameter optimization.
- Techniques: GridSearchCV and RandomizedSearchCV.
- Early stopping for iterative models.

#### Problems:
1. Tune the hyperparameters of a decision tree using GridSearchCV.
2. Optimize a random forest classifier using RandomizedSearchCV.
3. Experiment with early stopping while training a gradient boosting model.
4. Compare the performance of tuned and untuned models.

---

### 4. Pipelines for Automation
#### Key Concepts:
- Automating preprocessing and model training with Scikit-learn Pipelines.
- Combining feature engineering, scaling, and modeling in a single pipeline.
- Custom transformers for domain-specific preprocessing.

#### Problems:
1. Create a pipeline that handles missing values, scales features, and trains a regression model.
2. Build a pipeline for a classification problem that includes encoding, scaling, and model tuning.
3. Develop a custom transformer for specific data transformations and integrate it into a pipeline.

---

## Mini-Projects

### Project 1: Automated ML Pipeline for House Prices
- **Objective**: Build an automated pipeline to preprocess data and predict house prices.
- **Dataset**: Housing dataset with features like `square_footage`, `bedrooms`, `location`, and `price`.
- **Tasks**:
  1. Handle missing values and encode categorical variables.
  2. Normalize features and create polynomial features.
  3. Train a regression model and evaluate using RMSE and R².

### Project 2: Classification Pipeline for Loan Eligibility
- **Objective**: Create an end-to-end pipeline for predicting loan eligibility.
- **Dataset**: Loan dataset with features like `applicant_income`, `credit_score`, and `loan_status`.
- **Tasks**:
  1. Preprocess data (impute missing values, encode categorical variables, scale numerical features).
  2. Train multiple models and tune hyperparameters.
  3. Evaluate the best model using confusion matrices and ROC-AUC.

### Project 3: Custom Pipeline for E-commerce Data
- **Objective**: Develop a pipeline to predict customer purchasing behavior.
- **Dataset**: E-commerce dataset with features like `product_views`, `time_spent`, and `purchase` (binary).
- **Tasks**:
  1. Handle missing data and create new interaction features.
  2. Build a pipeline to preprocess data, train a classifier, and tune hyperparameters.
  3. Deploy the pipeline to make predictions on new data.

---

## Tools
- **Python**: Core programming language.
- **Scikit-learn**: For pipelines, feature engineering, and model tuning.
- **Pandas**: For data manipulation.
- **Matplotlib/Seaborn**: For visualizing evaluation metrics.

---

## Outcomes
By completing these tasks, you will:
1. Master feature engineering and preprocessing techniques.
2. Understand model evaluation and diagnostics for regression and classification.
3. Build and deploy automated pipelines for real-world problems.
4. Complete at least **3 mini-projects** to showcase your advanced ML skills.

---

Let me know if you need further guidance or additional resources to complete these tasks!
