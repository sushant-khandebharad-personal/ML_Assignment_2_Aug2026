import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Configuration
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
# Load training data
# ============================================================

@st.cache_data
def load_training_data():

    df = pd.read_csv(
        "data/adult.data",
        names=COLUMNS,
        skipinitialspace=True
    )

    df = df.replace("?", pd.NA)

    return df


# ============================================================
# Create preprocessing pipeline
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

    return ColumnTransformer([
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


# ============================================================
# Create selected model
# ============================================================

def create_model(model_name):

    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            random_state=42
        )

    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5
        )

    if model_name == "Naive Bayes":
        return GaussianNB()

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )


# ============================================================
# Train model and predict
# ============================================================

def run_model(model_name, test_df):

    training_df = load_training_data()

    X_train = training_df.drop(
        "income",
        axis=1
    )

    y_train = training_df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    X_test = test_df.drop(
        "income",
        axis=1
    )

    y_test = test_df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    model = create_model(model_name)

    model.fit(
        X_train_processed,
        y_train
    )

    predictions = model.predict(
        X_test_processed
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "AUC": roc_auc_score(
            y_test,
            probabilities
        ),
        "Precision": precision_score(
            y_test,
            predictions
        ),
        "Recall": recall_score(
            y_test,
            predictions
        ),
        "F1 Score": f1_score(
            y_test,
            predictions
        ),
        "MCC": matthews_corrcoef(
            y_test,
            predictions
        )
    }

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return metrics, matrix


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Income Classification")

st.write(
    "Compare classification models for predicting "
    "whether annual income exceeds $50K."
)

st.subheader("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

st.subheader("2. Select Model")

model_name = st.selectbox(
    "Choose a classification model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)


if uploaded_file is not None:

    test_df = pd.read_csv(
        uploaded_file
    )

    st.write(
        f"Uploaded dataset: {test_df.shape[0]} rows"
    )

    if st.button("Run Prediction"):

        with st.spinner(
            "Training model and evaluating test data..."
        ):

            metrics, matrix = run_model(
                model_name,
                test_df
            )

        st.subheader(
            f"Results — {model_name}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.4f}"
        )

        col2.metric(
            "AUC",
            f"{metrics['AUC']:.4f}"
        )

        col3.metric(
            "Precision",
            f"{metrics['Precision']:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{metrics['Recall']:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.4f}"
        )

        col6.metric(
            "MCC",
            f"{metrics['MCC']:.4f}"
        )

        st.subheader("Confusion Matrix")

        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual <=50K", "Actual >50K"],
            columns=["Predicted <=50K", "Predicted >50K"]
        )

        st.dataframe(
            confusion_df
        )

else:

    st.info(
        "Upload test_data.csv to begin."
    )