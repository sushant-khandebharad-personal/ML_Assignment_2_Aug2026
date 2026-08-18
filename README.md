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
- Official Adult test instances: 16,281
- Assignment test CSV: 500 rows
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
https://github.com/sushant-khandebharad-personal/ML_Assignment_2_Aug2026

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


| ML Model            |   Accuracy |    AUC | Precision | Recall |         F1 |        MCC |
| ------------------- | ---------: | -----: | --------: | -----: | ---------: | ---------: |
| Logistic Regression |     0.8558 | 0.9078 |    0.7406 | 0.6173 |     0.6734 |     0.5858 |
| Decision Tree       |     0.8104 | 0.7525 |    0.5993 | 0.6409 |     0.6194 |     0.4938 |
| KNN                 |     0.8366 | 0.8679 |    0.6775 | 0.6135 |     0.6439 |     0.5394 |
| Naive Bayes         |     0.5382 | 0.7495 |    0.3374 | 0.9528 |     0.4983 |     0.3294 |
| Random Forest       | **0.8575** | 0.9068 |    0.7356 | 0.6371 | **0.6828** | **0.5941** |

> **Note:** The model comparison metrics above are calculated on the 20% validation set from the 32,561 training instances. The Streamlit application separately evaluates the uploaded 500-row `test_data.csv`, so its displayed metrics may differ from the validation results above.

---

## 5. Model Performance Observations

### Logistic Regression

Logistic Regression performed very well, achieving 85.58% accuracy and the highest AUC of 90.78%. It also achieved the highest precision among the five models. Its recall was slightly lower than Decision Tree and Random Forest, but its overall performance was strong and balanced.


### Decision Tree

Decision Tree achieved 81.04% accuracy and an AUC of 75.25%, which were lower than Logistic Regression, KNN and Random Forest. However, it achieved a recall of 64.09%, which was higher than Logistic Regression and KNN. Its F1 score and MCC were lower than those of Logistic Regression and Random Forest.

### KNN

KNN achieved 83.66% accuracy and 86.79% AUC. Its performance was better than Decision Tree and Naive Bayes on most metrics, but lower than Logistic Regression and Random Forest. Its F1 score was 64.39% and MCC was 0.5394.


### Naive Bayes

Naive Bayes achieved a very high recall of 95.28%, meaning that it identified most of the actual `>50K` cases. However, its precision was only 33.74%, resulting in many false positives. Consequently, it achieved the lowest accuracy, F1 score and MCC among the evaluated models.

### Random Forest

Random Forest achieved the highest accuracy of 85.75% and the highest F1 score of 68.28% and MCC of 0.5941. Its AUC of 90.68% was also very strong. Overall, Random Forest provided the best balance across the evaluated metrics.


### Overall Winner

**Random Forest** is selected as the overall winner.

Random Forest achieved the highest accuracy (0.8575), F1 score (0.6828) and MCC (0.5941). Logistic Regression achieved the highest AUC (0.9078) and precision (0.7406), but Random Forest provided the strongest overall balance across the evaluated metrics.

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

  pip install -r requirements.txt
  
  streamlit run app.py

  http://localhost:8501


### Live Streamlit Application

    https://mlassignment2aug2026-ydliajpljrztue28rnmjog.streamlit.app/
