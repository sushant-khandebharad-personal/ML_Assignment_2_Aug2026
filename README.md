# Adult Income Classification - Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to build classification models that predict whether a person's annual income is greater than $50K based on demographic, employment, education and financial attributes.

Multiple machine learning classification algorithms are implemented and evaluated using the same dataset. The models are also integrated into an interactive Streamlit application where users can upload test data, select a model and view its evaluation results.

---

## 2. Dataset Description

The project uses the Adult Census Income dataset.

The dataset contains demographic and employment-related information used to predict the income category of an individual.

### Dataset characteristics

- Training instances: 32,561
- Official test instances: 16,281
- Input features: 14
- Target variable: `income`
- Classification type: Binary classification
- Target classes:
  - `<=50K`
  - `>50K`

### Features

The dataset contains the following attributes:

- age
- workclass
- fnlwgt
- education
- education_num
- marital_status
- occupation
- relationship
- race
- sex
- capital_gain
- capital_loss
- hours_per_week
- native_country

### Preprocessing

The following preprocessing steps were performed:

- Missing values represented by `?` were treated as missing values.
- Numerical missing values were imputed using the median.
- Categorical missing values were imputed using the most frequent value.
- Categorical features were converted using one-hot encoding.
- Numerical features were standardized using StandardScaler.
- The resulting processed dataset contained 105 features.

The training dataset was divided into training and validation sets using an 80:20 stratified split.

---

## 3. GitHub Repository Link

**Repository:**  
[To be updated after GitHub repository creation]

---

## 4. Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The following evaluation metrics were calculated for every model:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8558 | 0.9078 | 0.7406 | 0.6173 | 0.6734 | 0.5858 |
| Decision Tree | 0.8142 | 0.7537 | 0.6091 | 0.6371 | 0.6228 | 0.4999 |
| KNN | 0.8366 | 0.8679 | 0.6775 | 0.6135 | 0.6439 | 0.5394 |
| Naive Bayes | 0.5382 | 0.7495 | 0.3374 | 0.9528 | 0.4983 | 0.3294 |
| Random Forest | 0.8558 | 0.9065 | 0.7342 | 0.6288 | 0.6774 | 0.5882 |

---

## 5. Model Performance Observations

### Logistic Regression

Logistic Regression performed very well, achieving 85.58% accuracy and the highest AUC of 90.78%. It also achieved the highest precision among the five models. Its recall was slightly lower than Decision Tree and Random Forest, but its overall performance was strong and balanced.

### Decision Tree

Decision Tree achieved 81.42% accuracy and an AUC of 75.37%, which were lower than Logistic Regression, KNN and Random Forest. However, it achieved a recall of 63.71%, which was higher than Logistic Regression and KNN. Its F1 score and MCC were lower than those of Logistic Regression and Random Forest.

### KNN

KNN achieved 83.66% accuracy and 86.79% AUC. Its performance was better than Decision Tree and Naive Bayes on most metrics, but lower than Logistic Regression and Random Forest. Its F1 score was 64.39% and MCC was 0.5394.

### Naive Bayes

Naive Bayes achieved a very high recall of 95.28%, meaning that it identified most of the actual `>50K` cases. However, its precision was only 33.74%, resulting in many false positives. Consequently, it achieved the lowest accuracy, F1 score and MCC among the evaluated models.

### Random Forest

Random Forest achieved 85.58% accuracy, matching Logistic Regression. It achieved the highest F1 score of 67.74% and the highest MCC of 0.5882. Its AUC of 90.65% was also very strong. Overall, Random Forest provided the best balance across the evaluated metrics.

### Overall Winner

**Random Forest** is selected as the overall winner.

It achieved the highest F1 score and MCC while matching Logistic Regression's highest accuracy. Logistic Regression achieved slightly higher AUC and precision, but Random Forest provided the strongest overall balance across the evaluation metrics.

---

## 6. Streamlit Application

The project includes an interactive Streamlit application with the following features:

- CSV test-data upload
- Classification model selection
- Evaluation metrics display
- Confusion matrix display
- Support for all five implemented models

The application is designed to evaluate uploaded test data using the selected classification model.

### Local execution

Create and activate the Python virtual environment and install dependencies:

```bash
pip install -r requirements.txt