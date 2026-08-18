import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. Dataset configuration
# ============================================================

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


# ============================================================
# 2. Load dataset
# ============================================================

def load_training_data():

    df = pd.read_csv(
        "data/adult.data",
        names=COLUMNS,
        skipinitialspace=True
    )

    # Convert '?' to actual missing values
    df = df.replace("?", pd.NA)

    return df


# ============================================================
# 3. Create preprocessing pipeline
# ============================================================

def create_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor


# ============================================================
# 4. Load and prepare data
# ============================================================

df = load_training_data()

X = df.drop("income", axis=1)

y = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})


# ============================================================
# 5. Train / validation split
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== DATASET ==========")
print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])

print("\n========== DATA SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_valid.shape[0])


# ============================================================
# 6. Preprocessing
# ============================================================

preprocessor = create_preprocessor()

X_train_processed = preprocessor.fit_transform(X_train)

X_valid_processed = preprocessor.transform(X_valid)


print("\n========== PREPROCESSING ==========")
print("Original feature count:", X.shape[1])
print("Processed feature count:", X_train_processed.shape[1])
print("Processed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_valid_processed.shape)


# ============================================================
# 7. Logistic Regression
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

y_pred = logistic_model.predict(X_valid_processed)

y_prob = logistic_model.predict_proba(
    X_valid_processed
)[:, 1]


# ============================================================
# 8. Evaluation metrics
# ============================================================

accuracy = accuracy_score(y_valid, y_pred)

auc = roc_auc_score(
    y_valid,
    y_prob
)

precision = precision_score(
    y_valid,
    y_pred
)

recall = recall_score(
    y_valid,
    y_pred
)

f1 = f1_score(
    y_valid,
    y_pred
)

mcc = matthews_corrcoef(
    y_valid,
    y_pred
)


# ============================================================
# 9. Decision Tree
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

decision_tree_model.fit(
    X_train_processed,
    y_train
)

dt_pred = decision_tree_model.predict(
    X_valid_processed
)

dt_prob = decision_tree_model.predict_proba(
    X_valid_processed
)[:, 1]


# ============================================================
# 10. Decision Tree evaluation
# ============================================================

dt_accuracy = accuracy_score(
    y_valid,
    dt_pred
)

dt_auc = roc_auc_score(
    y_valid,
    dt_prob
)

dt_precision = precision_score(
    y_valid,
    dt_pred
)

dt_recall = recall_score(
    y_valid,
    dt_pred
)

dt_f1 = f1_score(
    y_valid,
    dt_pred
)

dt_mcc = matthews_corrcoef(
    y_valid,
    dt_pred
)


print("\n========== DECISION TREE ==========")

print(f"Accuracy  : {dt_accuracy:.4f}")
print(f"AUC       : {dt_auc:.4f}")
print(f"Precision : {dt_precision:.4f}")
print(f"Recall    : {dt_recall:.4f}")
print(f"F1 Score  : {dt_f1:.4f}")
print(f"MCC       : {dt_mcc:.4f}")




print("\n========== LOGISTIC REGRESSION ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"MCC       : {mcc:.4f}")



# ============================================================
# 11. K-Nearest Neighbors
# ============================================================

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_processed,
    y_train
)

knn_pred = knn_model.predict(
    X_valid_processed
)

knn_prob = knn_model.predict_proba(
    X_valid_processed
)[:, 1]


# ============================================================
# 12. KNN evaluation
# ============================================================

knn_accuracy = accuracy_score(
    y_valid,
    knn_pred
)

knn_auc = roc_auc_score(
    y_valid,
    knn_prob
)

knn_precision = precision_score(
    y_valid,
    knn_pred
)

knn_recall = recall_score(
    y_valid,
    knn_pred
)

knn_f1 = f1_score(
    y_valid,
    knn_pred
)

knn_mcc = matthews_corrcoef(
    y_valid,
    knn_pred
)


print("\n========== KNN ==========")

print(f"Accuracy  : {knn_accuracy:.4f}")
print(f"AUC       : {knn_auc:.4f}")
print(f"Precision : {knn_precision:.4f}")
print(f"Recall    : {knn_recall:.4f}")
print(f"F1 Score  : {knn_f1:.4f}")
print(f"MCC       : {knn_mcc:.4f}")


# ============================================================
# 13. Naive Bayes
# ============================================================

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(
    X_train_processed,
    y_train
)

nb_pred = naive_bayes_model.predict(
    X_valid_processed
)

nb_prob = naive_bayes_model.predict_proba(
    X_valid_processed
)[:, 1]


# ============================================================
# 14. Naive Bayes evaluation
# ============================================================

nb_accuracy = accuracy_score(
    y_valid,
    nb_pred
)

nb_auc = roc_auc_score(
    y_valid,
    nb_prob
)

nb_precision = precision_score(
    y_valid,
    nb_pred
)

nb_recall = recall_score(
    y_valid,
    nb_pred
)

nb_f1 = f1_score(
    y_valid,
    nb_pred
)

nb_mcc = matthews_corrcoef(
    y_valid,
    nb_pred
)


print("\n========== NAIVE BAYES ==========")

print(f"Accuracy  : {nb_accuracy:.4f}")
print(f"AUC       : {nb_auc:.4f}")
print(f"Precision : {nb_precision:.4f}")
print(f"Recall    : {nb_recall:.4f}")
print(f"F1 Score  : {nb_f1:.4f}")
print(f"MCC       : {nb_mcc:.4f}")



# ============================================================
# 15. Random Forest
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

rf_pred = random_forest_model.predict(
    X_valid_processed
)

rf_prob = random_forest_model.predict_proba(
    X_valid_processed
)[:, 1]


# ============================================================
# 16. Random Forest evaluation
# ============================================================

rf_accuracy = accuracy_score(
    y_valid,
    rf_pred
)

rf_auc = roc_auc_score(
    y_valid,
    rf_prob
)

rf_precision = precision_score(
    y_valid,
    rf_pred
)

rf_recall = recall_score(
    y_valid,
    rf_pred
)

rf_f1 = f1_score(
    y_valid,
    rf_pred
)

rf_mcc = matthews_corrcoef(
    y_valid,
    rf_pred
)


print("\n========== RANDOM FOREST ==========")

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"AUC       : {rf_auc:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"MCC       : {rf_mcc:.4f}")



# ============================================================
# 17. Model comparison
# ============================================================

results = {
    "Logistic Regression": [
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ],
    "Decision Tree": [
        dt_accuracy,
        dt_auc,
        dt_precision,
        dt_recall,
        dt_f1,
        dt_mcc
    ],
    "KNN": [
        knn_accuracy,
        knn_auc,
        knn_precision,
        knn_recall,
        knn_f1,
        knn_mcc
    ],
    "Naive Bayes": [
        nb_accuracy,
        nb_auc,
        nb_precision,
        nb_recall,
        nb_f1,
        nb_mcc
    ],
    "Random Forest": [
        rf_accuracy,
        rf_auc,
        rf_precision,
        rf_recall,
        rf_f1,
        rf_mcc
    ]
}

results_df = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
).T

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4))

results_df.to_csv(
    "model_comparison.csv"
)

print("\nModel comparison saved to model_comparison.csv")